import logging
import re
from typing import Optional, List
from urllib.parse import urlparse

import httpx

from config import get_settings
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


def _is_valid_dach_phone(number: str) -> bool:
    """
    Check if phone number is a valid DACH phone number.
    Accepts:
    - International format: +49, +43, +41, 0049, 0043, 0041
    - National format: 0xxx (German domestic, e.g. 0176, 030, 089)

    Filters out numbers that are too short or have no recognizable prefix.
    """
    if not number:
        return False

    # Clean the number - keep only digits and +
    cleaned = re.sub(r'[^\d+]', '', number)

    # Minimum length check (at least 8 digits for a real phone number)
    digits_only = re.sub(r'\D', '', cleaned)
    if len(digits_only) < 8:
        return False

    # Valid patterns:
    # 1. International DACH: +49, +43, +41
    if cleaned.startswith(('+49', '+43', '+41')):
        return True

    # 2. International with 00: 0049, 0043, 0041
    if cleaned.startswith(('0049', '0043', '0041')):
        return True

    # 3. German national format: starts with 0 (e.g. 0176, 030, 089)
    #    This covers mobile (015x, 016x, 017x) and landline (0xx)
    if cleaned.startswith('0') and not cleaned.startswith('00'):
        return True

    # No valid prefix found
    return False


async def enrich_lead(
    payload: WebhookPayload,
    skip_paid_apis: bool = False
) -> EnrichmentResult:
    """
    Main enrichment pipeline - OPTIMIZED FLOW v3.

    Flow:
    1.  LLM Parse → Extract contact name, company, email from job posting
    1b. Google Domain Search → If no domain, find via Google (FREE!)
    2.  Impressum Scraping (ALWAYS!) → Company phone, website, address (FREE!)
    3.  Decision Maker Discovery → Find executives (FREE!)
    4.  Google LinkedIn Search → Find personal LinkedIn URL (FREE!)
    5.  FullEnrich → Get PERSONAL mobile phone (PAID: 10 credits)
    6.  Kaspr → Fallback for mobile (PAID: 1 credit)
    7.  Company Research → Industry, employee count, sales brief (FREE!)
    8.  Company LinkedIn → Find company LinkedIn page (FREE!)

    OUTPUT - company object contains:
    - domain, industry, employee_count, location
    - phone (company landline), website, linkedin_url

    IMPORTANT:
    - Impressum = COMPANY data (main phone, address, website)
    - FullEnrich/Kaspr = PERSONAL data (decision maker's mobile)
    - Both are needed!

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
        location=parsed.location or payload.location  # Use job location as fallback
    )

    # Step 1b: Google Domain Search - If no domain from LLM, find it (FREE!)
    if not company_info.domain and parsed.company_name:
        logger.info("No domain from LLM - searching via Google...")
        found_domain = await _google_find_domain(parsed.company_name)
        if found_domain:
            company_info.domain = found_domain
            enrichment_path.append("google_domain_found")
            logger.info(f"Google found domain: {found_domain}")

    # Step 2: Impressum Scraping - ALWAYS run for company data (FREE!)
    # This gets us: company phone, website, address
    logger.info("Scraping Impressum for company data (always, not just fallback)...")
    impressum = ImpressumScraper()
    impressum_result = await impressum.scrape(
        company_name=parsed.company_name,
        domain=parsed.company_domain
    )

    if impressum_result:
        enrichment_path.append("impressum")

        # Store COMPANY phone (not personal!) - no filtering for company phones
        if impressum_result.phones:
            company_info.phone = impressum_result.phones[0].number
            enrichment_path.append("impressum_company_phone")
            logger.info(f"Impressum found company phone: {company_info.phone}")

        # Store website
        if impressum_result.website_url:
            company_info.website = impressum_result.website_url
            enrichment_path.append("impressum_website")
            logger.info(f"Impressum found website: {company_info.website}")

        # Store address as location (if not already set)
        if impressum_result.address and not company_info.location:
            company_info.location = impressum_result.address
            enrichment_path.append("impressum_address")
            logger.info(f"Impressum found address: {company_info.location}")

        logger.info(f"Impressum: {len(impressum_result.phones)} phones, {len(impressum_result.emails)} emails")

    # Step 3: Build decision maker from parsed contact
    decision_maker: Optional[DecisionMaker] = None
    collected_emails: List[str] = []
    linkedin_url: Optional[str] = None

    # Add Impressum emails to collection
    if impressum_result and impressum_result.emails:
        collected_emails.extend(impressum_result.emails)

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

    # Step 3b: Decision Maker Discovery - If NO contact in job posting, find one!
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

    # Step 4: Google LinkedIn Search - Find LinkedIn profile (FREE!)
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

    # Step 5: FullEnrich - Try to get PERSONAL phone
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

    # Step 6: Kaspr - Try if no PERSONAL phone OR only landline found
    # Kaspr: 1 credit per request, UNLIMITED emails
    # We want mobile numbers, so try Kaspr even if we have a landline
    need_kaspr = (
        not phone_result or
        (phone_result and phone_result.type == PhoneType.LANDLINE)
    )

    if not skip_paid_apis and need_kaspr and linkedin_url:
        reason = "no phone yet" if not phone_result else "only landline found, trying for mobile"
        logger.info(f"Trying Kaspr ({reason}) with LinkedIn: {linkedin_url}")
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
                kaspr_phone = _get_best_phone(kaspr_result.phones)
                # If Kaspr found mobile and we only had landline, prefer mobile
                if kaspr_phone:
                    if not phone_result:
                        phone_result = kaspr_phone
                        enrichment_path.append("kaspr_phone_found")
                    elif kaspr_phone.type == PhoneType.MOBILE and phone_result.type != PhoneType.MOBILE:
                        logger.info(f"Kaspr found mobile, replacing landline")
                        phone_result = kaspr_phone
                        enrichment_path.append("kaspr_mobile_upgrade")
                    logger.info(f"Kaspr found phone: {kaspr_phone.number} ({kaspr_phone.type.value})")

            logger.info(f"Kaspr: {len(kaspr_result.emails)} emails, {len(kaspr_result.phones)} phones")

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

            # Transfer research data to company_info
            if intel_result.industry and not company_info.industry:
                company_info.industry = intel_result.industry
            if intel_result.employee_count and not company_info.employee_count:
                company_info.employee_count = intel_result.employee_count
            if intel_result.headquarters and not company_info.location:
                company_info.location = intel_result.headquarters
    except Exception as e:
        logger.warning(f"Company research failed: {e}")

    # Step 8: Find Company LinkedIn URL (FREE!)
    if not company_info.linkedin_url and parsed.company_name:
        logger.info("Searching for company LinkedIn page...")
        company_linkedin = await _google_find_company_linkedin(
            parsed.company_name,
            company_info.domain
        )
        if company_linkedin:
            company_info.linkedin_url = company_linkedin
            enrichment_path.append("company_linkedin_found")
            logger.info(f"Found company LinkedIn: {company_linkedin}")

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
    # Success if we have: personal phone OR company phone OR emails
    success = (
        phone_result is not None or
        company_info.phone is not None or
        len(unique_emails) > 0
    )

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

    # Filter out phones without valid DACH prefix
    valid_phones = [p for p in phones if _is_valid_dach_phone(p.number)]

    if not valid_phones:
        logger.info("No phones with valid DACH prefix found - filtered out")
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

    return min(valid_phones, key=score)


async def _google_find_domain(company_name: str) -> Optional[str]:
    """
    Find company domain via Google Custom Search (FREE!).
    Searches for company website and extracts domain.
    """
    settings = get_settings()

    if not settings.google_api_key or not settings.google_cse_id:
        logger.warning("Google API keys not configured - skipping domain search")
        return None

    async with httpx.AsyncClient(timeout=settings.api_timeout) as client:
        # Search for company website
        query = f'"{company_name}" official website'
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": settings.google_api_key,
            "cx": settings.google_cse_id,
            "q": query,
            "num": 5
        }

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])

            # Skip common non-company domains
            skip_domains = {
                'linkedin.com', 'xing.com', 'facebook.com', 'twitter.com',
                'instagram.com', 'youtube.com', 'wikipedia.org', 'kununu.de',
                'glassdoor.com', 'indeed.com', 'stepstone.de', 'monster.de',
                'arbeitsagentur.de', 'meinestadt.de', 'gelbeseiten.de'
            }

            company_lower = company_name.lower().replace(" ", "").replace("-", "")

            for item in items:
                link = item.get("link", "")
                if not link:
                    continue

                parsed = urlparse(link)
                domain = parsed.netloc.replace("www.", "")

                # Skip social media and job portals
                if any(skip in domain for skip in skip_domains):
                    continue

                # Prefer domains that contain company name (without spaces)
                domain_clean = domain.lower().replace("-", "").replace(".", "")
                if company_lower[:4] in domain_clean:
                    logger.info(f"Found matching domain: {domain}")
                    return domain

            # If no matching domain, return first non-skipped result
            for item in items:
                link = item.get("link", "")
                if link:
                    parsed = urlparse(link)
                    domain = parsed.netloc.replace("www.", "")
                    if not any(skip in domain for skip in skip_domains):
                        return domain

        except Exception as e:
            logger.warning(f"Google domain search failed: {e}")

    return None


async def _google_find_company_linkedin(
    company_name: str,
    domain: Optional[str] = None
) -> Optional[str]:
    """
    Find company LinkedIn page via Google Custom Search (FREE!).
    Returns LinkedIn company URL.
    """
    settings = get_settings()

    if not settings.google_api_key or not settings.google_cse_id:
        return None

    async with httpx.AsyncClient(timeout=settings.api_timeout) as client:
        # Search for company LinkedIn page
        query = f'"{company_name}" site:linkedin.com/company'

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": settings.google_api_key,
            "cx": settings.google_cse_id,
            "q": query,
            "num": 3
        }

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])

            for item in items:
                link = item.get("link", "")
                # Must be a LinkedIn company page
                if "linkedin.com/company/" in link:
                    # Clean up URL
                    linkedin_url = link.split("?")[0]  # Remove query params
                    # Normalize to https
                    if linkedin_url.startswith("http://"):
                        linkedin_url = linkedin_url.replace("http://", "https://")
                    return linkedin_url

        except Exception as e:
            logger.warning(f"Google company LinkedIn search failed: {e}")

    return None
