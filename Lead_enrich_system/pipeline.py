import logging
from typing import Optional, List

from models import (
    WebhookPayload, EnrichmentResult, CompanyInfo, CompanyIntel,
    DecisionMaker, PhoneResult, PhoneSource, PhoneType
)
from llm_parser import parse_job_posting
from clients.kaspr import KasprClient
from clients.fullenrich import FullEnrichClient
from clients.impressum import ImpressumScraper
from clients.linkedin_search import LinkedInSearchClient
from clients.company_research import CompanyResearcher

logger = logging.getLogger(__name__)


async def enrich_lead(
    payload: WebhookPayload,
    skip_paid_apis: bool = False
) -> EnrichmentResult:
    """
    Main enrichment pipeline - OPTIMIZED FLOW v2.

    Flow:
    1. LLM Parse → Extract contact name, company, email from job posting
    2. Decision Maker Discovery → If no contact found, search for executives
       - Apollo Free Search (no credits!)
       - Google Decision Maker Search (FREE!)
    3. Google LinkedIn Search → Find LinkedIn URL (FREE with Google API)
    4. FullEnrich → Try to get phone FIRST (10 credits/phone, but saves Kaspr)
    5. Kaspr (fallback) → If no phone yet + have LinkedIn (1 credit)
    6. Impressum Scraping → Free fallback for company phone

    Args:
        payload: Job posting data
        skip_paid_apis: If True, skip all paid API calls (for testing)
    """
    enrichment_path = []

    # Step 1: Parse job posting with Claude
    logger.info(f"=== Starting enrichment for: {payload.company} ===")
    enrichment_path.append("llm_parse")

    parsed = await parse_job_posting(payload)
    logger.info(f"LLM extracted: domain={parsed.company_domain}, contact={parsed.contact_name}, email={parsed.contact_email}")

    # Create company info from parsed data
    company_info = CompanyInfo(
        name=parsed.company_name,
        domain=parsed.company_domain,
        location=parsed.location
    )

    # Step 2: Build decision maker from parsed contact
    decision_maker: Optional[DecisionMaker] = None
    collected_emails: List[str] = []
    linkedin_url: Optional[str] = None

    if parsed.contact_name:
        names = parsed.contact_name.split()
        first_name = names[0] if names else ""
        last_name = " ".join(names[1:]) if len(names) > 1 else ""

        decision_maker = DecisionMaker(
            name=parsed.contact_name,
            first_name=first_name,
            last_name=last_name,
            email=parsed.contact_email
        )

        if parsed.contact_email:
            collected_emails.append(parsed.contact_email)

        enrichment_path.append("contact_from_posting")
        logger.info(f"Contact from posting: {parsed.contact_name}")

    # Step 2b: Decision Maker Discovery - If NO contact in job posting, find one!
    # Priority: HR/Recruiting > Job-relevant dept head > General executive
    if not decision_maker and parsed.company_name:
        logger.info("No contact in job posting - searching for decision maker...")
        logger.info(f"Job category: {payload.category}")

        # Google Decision Maker Search (FREE!) - with job category for smart prioritization
        linkedin_client = LinkedInSearchClient()

        dm_result = await linkedin_client.find_decision_maker(
            company=parsed.company_name,
            domain=parsed.company_domain,
            job_category=payload.category  # Pass category for relevant dept head search
        )

        if dm_result:
            names = dm_result["name"].split()
            first_name = names[0] if names else ""
            last_name = " ".join(names[1:]) if len(names) > 1 else ""

            decision_maker = DecisionMaker(
                name=dm_result["name"],
                first_name=first_name,
                last_name=last_name,
                title=dm_result.get("title"),
                linkedin_url=dm_result.get("linkedin_url")
            )
            linkedin_url = dm_result.get("linkedin_url")
            enrichment_path.append("google_dm_search")
            logger.info(f"Google found decision maker: {dm_result['name']} ({dm_result.get('title')})")

    # Step 3: Google LinkedIn Search - Find LinkedIn profile (FREE!)
    # Only if we have a name but no LinkedIn URL yet
    if decision_maker and not linkedin_url:
        logger.info("Searching LinkedIn via Google...")
        linkedin_client = LinkedInSearchClient()

        found_url = await linkedin_client.find_linkedin_profile(
            name=decision_maker.name,
            company=parsed.company_name,
            domain=parsed.company_domain
        )

        if found_url:
            linkedin_url = found_url
            decision_maker.linkedin_url = linkedin_url
            enrichment_path.append("google_linkedin_found")
            logger.info(f"Google found LinkedIn: {linkedin_url}")

    # Step 4: FullEnrich FIRST - Try to get phone before using Kaspr
    # FullEnrich: 10 credits/phone, but saves Kaspr credits
    phone_result: Optional[PhoneResult] = None

    if not skip_paid_apis and decision_maker and decision_maker.first_name and decision_maker.last_name:
        logger.info("Trying FullEnrich FIRST (saves Kaspr credits)...")
        fullenrich = FullEnrichClient()

        fe_result = await fullenrich.enrich(
            first_name=decision_maker.first_name,
            last_name=decision_maker.last_name,
            company_name=parsed.company_name,
            domain=parsed.company_domain,
            linkedin_url=linkedin_url
        )

        if fe_result:
            enrichment_path.append("fullenrich")
            collected_emails.extend(fe_result.emails)

            # Check if FullEnrich found LinkedIn (if we didn't have it)
            if not linkedin_url and fe_result.linkedin_url:
                linkedin_url = fe_result.linkedin_url
                decision_maker.linkedin_url = linkedin_url
                enrichment_path.append("fullenrich_linkedin_found")
                logger.info(f"FullEnrich found LinkedIn: {linkedin_url}")

            # Check if FullEnrich found phones
            if fe_result.phones:
                phone_result = _get_best_phone(fe_result.phones)
                enrichment_path.append("fullenrich_phone_found")
                logger.info(f"FullEnrich found phone: {phone_result.number}")

            logger.info(f"FullEnrich: {len(fe_result.emails)} emails, {len(fe_result.phones)} phones")

    # Step 5: Kaspr - FALLBACK if no phone yet and we have LinkedIn
    # Kaspr: 1 credit per request, UNLIMITED emails
    if not skip_paid_apis and not phone_result and linkedin_url:
        logger.info(f"No phone yet - trying Kaspr with LinkedIn: {linkedin_url}")
        kaspr = KasprClient()

        kaspr_result = await kaspr.enrich_by_linkedin(
            linkedin_url=linkedin_url,
            name=decision_maker.name if decision_maker else ""
        )

        if kaspr_result:
            enrichment_path.append("kaspr")
            # Kaspr emails are unlimited - always collect
            collected_emails.extend(kaspr_result.emails)

            if kaspr_result.phones:
                phone_result = _get_best_phone(kaspr_result.phones)
                enrichment_path.append("kaspr_phone_found")
                logger.info(f"Kaspr found phone: {phone_result.number} ({phone_result.type.value})")

            logger.info(f"Kaspr: {len(kaspr_result.emails)} emails, {len(kaspr_result.phones)} phones")

    # Step 6: Impressum Scraping - FREE fallback for company phone
    if not phone_result:
        logger.info("Trying Impressum scraping...")
        impressum = ImpressumScraper()

        imp_result = await impressum.scrape(
            company_name=parsed.company_name,
            domain=parsed.company_domain
        )

        if imp_result:
            enrichment_path.append("impressum")
            collected_emails.extend(imp_result.emails)

            if imp_result.phones:
                phone_result = _get_best_phone(imp_result.phones)
                enrichment_path.append("impressum_phone_found")
                logger.info(f"Impressum found phone: {phone_result.number}")

            logger.info(f"Impressum: {len(imp_result.emails)} emails, {len(imp_result.phones)} phones")

    # Step 7: Company Research - FREE sales intelligence
    logger.info("Researching company for sales brief...")
    company_intel: Optional[CompanyIntel] = None

    try:
        researcher = CompanyResearcher()
        intel_result = await researcher.research(
            company_name=parsed.company_name,
            domain=parsed.company_domain,
            job_description=payload.description,
            job_title=payload.title
        )

        if intel_result and intel_result.summary:
            company_intel = CompanyIntel(
                summary=intel_result.summary,
                description=intel_result.description,
                industry=intel_result.industry,
                employee_count=intel_result.employee_count,
                founded=intel_result.founded,
                headquarters=intel_result.headquarters,
                products_services=intel_result.products_services,
                hiring_signals=intel_result.hiring_signals,
                website_url=intel_result.website_url
            )
            enrichment_path.append("company_research")
            logger.info(f"Company research complete: {len(intel_result.summary)} char summary")
    except Exception as e:
        logger.warning(f"Company research failed: {e}")

    # Deduplicate and clean emails
    unique_emails = list(set(e.lower().strip() for e in collected_emails if e and '@' in e))

    # Update decision maker email - prefer personal email
    if decision_maker and unique_emails:
        personal_emails = [e for e in unique_emails if not e.startswith(('kontakt@', 'info@', 'contact@', 'bewerbung@'))]
        if personal_emails:
            decision_maker.email = personal_emails[0]
        elif not decision_maker.email:
            decision_maker.email = unique_emails[0]

    # Build result
    success = phone_result is not None or len(unique_emails) > 0

    result = EnrichmentResult(
        success=success,
        company=company_info,
        company_intel=company_intel,
        decision_maker=decision_maker,
        phone=phone_result,
        emails=unique_emails,
        enrichment_path=enrichment_path,
        job_id=payload.id,
        job_title=payload.title
    )

    logger.info(f"=== Enrichment complete: success={success}, path={' -> '.join(enrichment_path)} ===")
    return result


async def enrich_lead_test_mode(payload: WebhookPayload) -> EnrichmentResult:
    """
    Test mode - only uses LLM parsing and free services (Impressum).
    NO paid API credits consumed.
    """
    return await enrich_lead(payload, skip_paid_apis=True)


def _get_best_phone(phones: List[PhoneResult]) -> Optional[PhoneResult]:
    """Select best phone from list, preferring mobile numbers."""
    if not phones:
        return None

    # Sort: mobile first, then by source priority
    source_priority = {
        PhoneSource.KASPR: 0,
        PhoneSource.FULLENRICH: 1,
        PhoneSource.IMPRESSUM: 2,
        PhoneSource.COMPANY_MAIN: 3
    }

    def score(p: PhoneResult) -> tuple:
        type_score = 0 if p.type == PhoneType.MOBILE else 1
        source_score = source_priority.get(p.source, 99)
        return (type_score, source_score)

    return min(phones, key=score)
