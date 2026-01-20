import logging
import re
from typing import Optional, List
from urllib.parse import urlparse

import httpx

from config import get_settings
from models import (
    WebhookPayload, EnrichmentResult, CompanyInfo, CompanyIntel,
    DecisionMaker, PhoneResult, PhoneSource, PhoneType, PhoneStatus
)
from llm_parser import parse_job_posting
from clients.kaspr import KasprClient
from clients.fullenrich import FullEnrichClient
from clients.impressum import ImpressumScraper
from clients.linkedin_search import LinkedInSearchClient
from clients.company_research import CompanyResearcher
from clients.job_scraper import JobUrlScraper, ScrapedContact
from utils.stats import track_phone_attempt

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


def _extract_name_from_email(email: str) -> Optional[str]:
    """
    Extract a real person name from email address.

    Examples:
        hans.hermann@amasol.de -> Hans Hermann
        mohit.popli@amasol.de -> Mohit Popli
        m.mueller@company.de -> None (too short)
        info@company.de -> None (generic)
    """
    if not email or '@' not in email:
        return None

    local_part = email.split('@')[0].lower()

    # Skip generic email addresses
    generic_patterns = [
        'info', 'kontakt', 'contact', 'mail', 'office', 'hello', 'team',
        'support', 'service', 'sales', 'marketing', 'hr', 'personal',
        'bewerbung', 'karriere', 'jobs', 'admin', 'webmaster', 'noreply'
    ]
    if local_part in generic_patterns:
        return None

    # Try to extract name parts
    # Pattern 1: firstname.lastname
    if '.' in local_part:
        parts = local_part.split('.')
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]

            # Skip if parts are too short (initials like m.mueller)
            if len(first) < 2 or len(last) < 2:
                return None

            # Skip if contains numbers
            if any(c.isdigit() for c in first + last):
                return None

            # Capitalize properly
            first_name = first.capitalize()
            last_name = last.capitalize()

            return f"{first_name} {last_name}"

    # Pattern 2: firstnamelastname (no separator) - harder to parse, skip
    return None


def _is_valid_person_name(name: str) -> bool:
    """
    Quick validation if a string looks like a real person name.
    Used to filter out HTML garbage before using as decision maker.
    """
    if not name:
        return False

    # Clean whitespace
    name = ' '.join(name.split())

    # Check for HTML artifacts
    if '\t' in name or '\n' in name:
        return False

    # Must have at least 2 words
    words = name.split()
    if len(words) < 2:
        return False

    # Length check (typical names are 5-40 chars)
    if len(name) < 5 or len(name) > 40:
        return False

    # Filter out job titles and garbage
    invalid_patterns = [
        'präsident', 'vizepräsident', 'vize',
        'teamleiter', 'abteilungsleiter', 'bereichsleiter', 'gruppenleiter',
        'geschäftsführ', 'geschäftsleitung',
        'vorstand', 'aufsichtsrat', 'beirat',
        'direktor', 'director', 'manager', 'leiter', 'leiterin',
        'chef', 'chefin', 'head of', 'senior', 'junior',
        'ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio',
        'und team', 'unser team', 'das team',
        'kontakt', 'email', 'telefon', 'gmbh', 'ag', 'kg',
    ]

    name_lower = name.lower()
    if any(p in name_lower for p in invalid_patterns):
        return False

    # Each word should be alphabetic and capitalized
    for word in words[:2]:
        clean = word.replace('-', '')
        if not clean.isalpha():
            return False
        if not word[0].isupper():
            return False

    return True


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

    # Step 1a: Scrape original job URL for contact person (if URL provided)
    scraped_contact: Optional[ScrapedContact] = None
    if payload.url:
        logger.info(f"Step 1a: Scraping job URL for contact: {payload.url}")
        try:
            job_scraper = JobUrlScraper(timeout=10)
            scraped_contact = await job_scraper.scrape_contact(payload.url)

            if scraped_contact:
                enrichment_path.append("job_url_scraped")
                logger.info(f"Job URL found contact: {scraped_contact.name} / {scraped_contact.email} (confidence: {scraped_contact.confidence:.2f})")

                # If we found a better contact than LLM parsing, use it
                if scraped_contact.name and not parsed.contact_name:
                    parsed.contact_name = scraped_contact.name
                    enrichment_path.append("contact_from_job_url")
                if scraped_contact.email and not parsed.contact_email:
                    parsed.contact_email = scraped_contact.email
                    enrichment_path.append("email_from_job_url")
            else:
                enrichment_path.append("job_url_no_contact")
        except Exception as e:
            logger.warning(f"Job URL scraping failed: {e}")
            enrichment_path.append("job_url_error")

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

    # Add emails from scraped job URL
    if scraped_contact and scraped_contact.email:
        collected_emails.append(scraped_contact.email)

    # Step 3: Build candidate list for phone enrichment
    # Priority: 1. Contact from job posting 2. Team page 3. LinkedIn search
    # Max 3 candidates total, stop when phone found
    phone_result: Optional[PhoneResult] = None
    dm_candidates = []

    # Step 3a: Contact from job posting = FIRST candidate (highest priority!)
    if parsed.contact_name:
        names = parsed.contact_name.split()
        first_name = names[0] if names else ""
        last_name = " ".join(names[1:]) if len(names) > 1 else ""

        dm_candidates.append({
            "name": parsed.contact_name,
            "title": None,
            "linkedin_url": None,
            "verified_current": True,  # From job posting = definitely current
            "source": "job_posting",
            "email": parsed.contact_email
        })

        if parsed.contact_email:
            collected_emails.append(parsed.contact_email)

        enrichment_path.append("contact_from_posting")
        logger.info(f"Contact from posting: {parsed.contact_name} (will try first)")

    # Step 3b: If we need more candidates (< 3), search for more
    if len(dm_candidates) < 3 and parsed.company_name:
        remaining_slots = 3 - len(dm_candidates)
        logger.info(f"Searching for {remaining_slots} more candidates (have {len(dm_candidates)} from posting)...")
        logger.info(f"Job category: {payload.category}")

        # STEP 3b-1: Try Team Page FIRST (verified contacts!)
        logger.info("Step 3b-1: Checking company team page for verified contacts...")
        impressum_scraper = ImpressumScraper()
        team_result = await impressum_scraper.scrape_team_page(
            company_name=parsed.company_name,
            domain=company_info.domain,
            job_category=payload.category
        )

        if team_result and team_result.members:
            logger.info(f"Found {len(team_result.members)} team members on company website (VERIFIED)")
            enrichment_path.append(f"team_page_{len(team_result.members)}_members")

            # Convert TeamMembers to candidate format (these are VERIFIED!)
            # Only add as many as we have remaining slots
            for member in team_result.members[:remaining_slots]:
                # Don't add if already have this person from job posting
                if not any(c["name"].lower() == member.name.lower() for c in dm_candidates):
                    dm_candidates.append({
                        "name": member.name,
                        "title": member.title,
                        "linkedin_url": None,  # Will search LinkedIn next
                        "verified_current": True,  # Team page = definitely current employee
                        "source": "team_page"
                    })

        # STEP 3b-2: If not enough from team page, supplement with LinkedIn search
        if len(dm_candidates) < 3:
            remaining = 3 - len(dm_candidates)
            logger.info(f"Step 3b-2: Searching LinkedIn for {remaining} more candidates...")

            linkedin_client = LinkedInSearchClient()
            linkedin_candidates = await linkedin_client.find_multiple_decision_makers(
                company=parsed.company_name,
                domain=parsed.company_domain,
                job_category=payload.category,
                max_candidates=remaining
            )

            # Add LinkedIn candidates (mark source)
            for lc in linkedin_candidates:
                # Don't add if we already have this person from team page
                if not any(c["name"].lower() == lc["name"].lower() for c in dm_candidates):
                    lc["source"] = "linkedin"
                    dm_candidates.append(lc)

        # Ensure max 3 candidates total
        dm_candidates = dm_candidates[:3]

    # Log total candidates
    enrichment_path.append(f"total_{len(dm_candidates)}_candidates")
    logger.info(f"Total candidates to try: {len(dm_candidates)}")

    # Step 4: Try each candidate until we find a phone (max 3!)
    if dm_candidates:
        for idx, dm_result in enumerate(dm_candidates):
            candidate_num = idx + 1
            source = dm_result.get("source", "unknown")
            logger.info(f"=== Trying candidate {candidate_num}/{len(dm_candidates)}: {dm_result['name']} (from {source}) ===")

            names = dm_result["name"].split()
            first_name = names[0] if names else ""
            last_name = " ".join(names[1:]) if len(names) > 1 else ""

            is_verified = dm_result.get("verified_current", False)
            verification_note = None if is_verified else "(nicht verifiziert - könnte nicht mehr dort arbeiten)"

            current_dm = DecisionMaker(
                name=dm_result["name"],
                first_name=first_name,
                last_name=last_name,
                title=dm_result.get("title"),
                linkedin_url=dm_result.get("linkedin_url"),
                verified_current=is_verified,
                verification_note=verification_note
            )

            # Add email from job posting if available
            if dm_result.get("email"):
                current_dm.email = dm_result["email"]

            current_linkedin_url = dm_result.get("linkedin_url")

            status = "VERIFIED" if is_verified else "UNVERIFIED"
            logger.info(f"Candidate {candidate_num} ({status}): {dm_result['name']} ({dm_result.get('title')})")

            # For candidates without LinkedIn URL: Search LinkedIn first
            if not current_linkedin_url and source in ["team_page", "job_posting"]:
                logger.info(f"Searching LinkedIn for: {dm_result['name']}")
                linkedin_client = LinkedInSearchClient()
                found_url = await linkedin_client.find_linkedin_profile(
                    name=dm_result["name"],
                    company=parsed.company_name,
                    domain=parsed.company_domain
                )
                if found_url:
                    current_linkedin_url = found_url
                    current_dm.linkedin_url = found_url
                    enrichment_path.append(f"linkedin_found_candidate_{candidate_num}")
                    logger.info(f"Found LinkedIn: {found_url}")

            # Try to get phone for this candidate
            candidate_phone, candidate_emails = await _try_enrich_candidate(
                candidate=current_dm,
                linkedin_url=current_linkedin_url,
                company_name=parsed.company_name,
                domain=parsed.company_domain,
                skip_paid_apis=skip_paid_apis,
                enrichment_path=enrichment_path
            )

            collected_emails.extend(candidate_emails)

            if candidate_phone:
                # Found a phone! Use this candidate
                phone_result = candidate_phone
                decision_maker = current_dm
                linkedin_url = current_linkedin_url
                enrichment_path.append(f"phone_found_candidate_{candidate_num}")
                logger.info(f"SUCCESS: Found phone for candidate {candidate_num}: {candidate_phone.number}")
                break
            else:
                logger.info(f"No phone found for candidate {candidate_num}, trying next...")

        # If no phone found with any candidate, use the first (best) one
        if not decision_maker and dm_candidates:
            dm_result = dm_candidates[0]
            names = dm_result["name"].split()
            first_name = names[0] if names else ""
            last_name = " ".join(names[1:]) if len(names) > 1 else ""
            is_verified = dm_result.get("verified_current", False)

            decision_maker = DecisionMaker(
                name=dm_result["name"],
                first_name=first_name,
                last_name=last_name,
                title=dm_result.get("title"),
                linkedin_url=dm_result.get("linkedin_url"),
                verified_current=is_verified,
                verification_note=None if is_verified else "(nicht verifiziert)"
            )
            if dm_result.get("email"):
                decision_maker.email = dm_result["email"]
            linkedin_url = dm_result.get("linkedin_url")
            enrichment_path.append("using_best_candidate_no_phone")
            logger.info(f"No phone found, using best candidate: {decision_maker.name}")

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

    # VALIDATION: Ensure decision maker has a valid real person name
    if decision_maker and not _is_valid_person_name(decision_maker.name):
        logger.warning(f"Decision maker name looks invalid: '{decision_maker.name}' - trying to extract from emails")
        enrichment_path.append("invalid_name_detected")

        # Try to extract name from collected emails
        extracted_name = None
        for email in unique_emails:
            extracted_name = _extract_name_from_email(email)
            if extracted_name:
                logger.info(f"Extracted name from email: {extracted_name}")
                break

        if extracted_name:
            # Update decision maker with extracted name
            names = extracted_name.split()
            decision_maker.name = extracted_name
            decision_maker.first_name = names[0] if names else ""
            decision_maker.last_name = " ".join(names[1:]) if len(names) > 1 else ""
            enrichment_path.append("name_from_email")
            logger.info(f"Updated decision maker name to: {extracted_name}")
        else:
            # No valid name found - clear the decision maker
            logger.warning("Could not extract valid name from emails - clearing decision maker")
            decision_maker = None
            enrichment_path.append("no_valid_name_found")

    # Build result
    # Success if we have: personal phone OR company phone OR emails
    success = (
        phone_result is not None or
        company_info.phone is not None or
        len(unique_emails) > 0
    )

    # Determine phone status for clear feedback
    if phone_result:
        if phone_result.type == PhoneType.MOBILE:
            phone_status = PhoneStatus.FOUND_MOBILE
        else:
            phone_status = PhoneStatus.FOUND_LANDLINE
    elif skip_paid_apis:
        phone_status = PhoneStatus.SKIPPED_PAID_API
    elif not decision_maker:
        phone_status = PhoneStatus.NO_DECISION_MAKER
    elif not linkedin_url:
        phone_status = PhoneStatus.NO_LINKEDIN
    elif "filtered_non_dach" in enrichment_path or any("filtered" in p for p in enrichment_path):
        phone_status = PhoneStatus.FILTERED_NON_DACH
    else:
        phone_status = PhoneStatus.API_NO_RESULT

    result = EnrichmentResult(
        success=success,
        company=company_info,
        company_intel=company_intel,
        decision_maker=decision_maker,
        phone=phone_result,
        phone_status=phone_status,
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


async def _try_enrich_candidate(
    candidate: DecisionMaker,
    linkedin_url: Optional[str],
    company_name: str,
    domain: Optional[str],
    skip_paid_apis: bool,
    enrichment_path: List[str]
) -> tuple[Optional[PhoneResult], List[str]]:
    """
    Try to get phone number for a single candidate using FullEnrich and Kaspr.

    Returns:
        Tuple of (PhoneResult or None, list of collected emails)
    """
    phone_result: Optional[PhoneResult] = None
    collected_emails: List[str] = []

    if skip_paid_apis:
        return None, []

    if not candidate.first_name or not candidate.last_name:
        logger.info(f"Skipping enrichment - incomplete name: {candidate.name}")
        return None, []

    # Try FullEnrich first (10 credits/phone)
    logger.info(f"Trying FullEnrich for {candidate.name}...")
    fullenrich = FullEnrichClient()

    fe_result = await fullenrich.enrich(
        first_name=candidate.first_name,
        last_name=candidate.last_name,
        company_name=company_name,
        domain=domain,
        linkedin_url=linkedin_url
    )

    if fe_result:
        enrichment_path.append("fullenrich")
        collected_emails.extend(fe_result.emails)

        # Check if FullEnrich found LinkedIn (if we didn't have it)
        if not linkedin_url and fe_result.linkedin_url:
            linkedin_url = fe_result.linkedin_url
            candidate.linkedin_url = linkedin_url
            enrichment_path.append("fullenrich_linkedin_found")
            logger.info(f"FullEnrich found LinkedIn: {linkedin_url}")

        # Check if FullEnrich found phones
        if fe_result.phones:
            fe_phone = _get_best_phone(fe_result.phones)
            if fe_phone:
                phone_result = fe_phone
                enrichment_path.append("fullenrich_phone_found")
                logger.info(f"FullEnrich found phone: {phone_result.number}")
                track_phone_attempt(
                    service="fullenrich",
                    phones_returned=fe_result.phones,
                    dach_valid_phone=fe_phone,
                    phone_type=fe_phone.type.value
                )
            else:
                logger.info(f"FullEnrich returned {len(fe_result.phones)} phones but none with valid DACH prefix")
                enrichment_path.append("fullenrich_filtered_non_dach")
                track_phone_attempt(
                    service="fullenrich",
                    phones_returned=fe_result.phones,
                    dach_valid_phone=None,
                    phone_type=None
                )
        else:
            track_phone_attempt(
                service="fullenrich",
                phones_returned=[],
                dach_valid_phone=None,
                phone_type=None
            )

        logger.info(f"FullEnrich: {len(fe_result.emails)} emails, {len(fe_result.phones)} phones")

    # Try Kaspr if no phone yet OR only landline found
    need_kaspr = (
        not phone_result or
        (phone_result and phone_result.type == PhoneType.LANDLINE)
    )

    if need_kaspr and linkedin_url:
        reason = "no phone yet" if not phone_result else "only landline, trying for mobile"
        logger.info(f"Trying Kaspr ({reason}) for {candidate.name}...")
        kaspr = KasprClient()

        kaspr_result = await kaspr.enrich_by_linkedin(
            linkedin_url=linkedin_url,
            name=candidate.name
        )

        if kaspr_result:
            enrichment_path.append("kaspr")
            collected_emails.extend(kaspr_result.emails)

            if kaspr_result.phones:
                kaspr_phone = _get_best_phone(kaspr_result.phones)
                if kaspr_phone:
                    if not phone_result:
                        phone_result = kaspr_phone
                        enrichment_path.append("kaspr_phone_found")
                    elif kaspr_phone.type == PhoneType.MOBILE and phone_result.type != PhoneType.MOBILE:
                        logger.info(f"Kaspr found mobile, replacing landline")
                        phone_result = kaspr_phone
                        enrichment_path.append("kaspr_mobile_upgrade")
                    logger.info(f"Kaspr found phone: {kaspr_phone.number} ({kaspr_phone.type.value})")
                    track_phone_attempt(
                        service="kaspr",
                        phones_returned=kaspr_result.phones,
                        dach_valid_phone=kaspr_phone,
                        phone_type=kaspr_phone.type.value
                    )
                else:
                    logger.info(f"Kaspr returned {len(kaspr_result.phones)} phones but none with valid DACH prefix")
                    enrichment_path.append("kaspr_filtered_non_dach")
                    track_phone_attempt(
                        service="kaspr",
                        phones_returned=kaspr_result.phones,
                        dach_valid_phone=None,
                        phone_type=None
                    )
            else:
                track_phone_attempt(
                    service="kaspr",
                    phones_returned=[],
                    dach_valid_phone=None,
                    phone_type=None
                )

            logger.info(f"Kaspr: {len(kaspr_result.emails)} emails, {len(kaspr_result.phones)} phones")

    return phone_result, collected_emails


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
        # Search for company LinkedIn page - include domain if available for better results
        if domain:
            query = f'"{company_name}" OR "{domain}" site:linkedin.com/company'
        else:
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
