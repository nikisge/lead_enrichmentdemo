import logging
import re
import httpx
from typing import Optional, List
from bs4 import BeautifulSoup
from dataclasses import dataclass

from config import get_settings
from models import PhoneResult, PhoneSource, PhoneType

logger = logging.getLogger(__name__)


@dataclass
class ImpressumResult:
    """Result from Impressum scraping."""
    phones: List[PhoneResult]
    emails: List[str]
    success: bool


class ImpressumScraper:
    """
    Scrapes Impressum pages for contact information.
    Free fallback when paid services don't find a phone.
    """

    def __init__(self):
        settings = get_settings()
        self.google_api_key = settings.google_api_key
        self.google_cse_id = settings.google_cse_id
        self.timeout = settings.api_timeout

    async def scrape(
        self,
        company_name: str,
        domain: Optional[str] = None
    ) -> Optional[ImpressumResult]:
        """
        Find and scrape company Impressum page.

        Strategy:
        1. If domain known, try {domain}/impressum directly
        2. Use Google to find Impressum page
        3. Scrape found page for phone/email
        """
        urls_to_try = []

        # Try direct URLs first
        if domain:
            clean_domain = domain.replace("www.", "")
            urls_to_try.extend([
                f"https://www.{clean_domain}/impressum",
                f"https://{clean_domain}/impressum",
                f"https://www.{clean_domain}/impressum.html",
                f"https://{clean_domain}/de/impressum",
                f"https://www.{clean_domain}/kontakt",
            ])

        # Try to scrape each URL
        for url in urls_to_try:
            result = await self._scrape_url(url)
            if result and result.success:
                return result

        # Fallback: Google search
        if self.google_api_key and self.google_cse_id:
            google_url = await self._google_search(company_name, domain)
            if google_url:
                result = await self._scrape_url(google_url)
                if result and result.success:
                    return result

        return None

    async def _google_search(
        self,
        company_name: str,
        domain: Optional[str]
    ) -> Optional[str]:
        """Use Google Custom Search to find Impressum page."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            query = f'"{company_name}" impressum'
            if domain:
                query += f" site:{domain}"

            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
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
                    # Prefer Impressum pages
                    if "impressum" in link.lower():
                        return link
                    # Or kontakt pages
                    if "kontakt" in link.lower():
                        return link

                # Return first result if no Impressum found
                if items:
                    return items[0].get("link")

            except Exception as e:
                logger.warning(f"Google search failed: {e}")

            return None

    async def _scrape_url(self, url: str) -> Optional[ImpressumResult]:
        """Scrape a URL for contact information."""
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        ) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "lxml")
                text = soup.get_text(separator=" ")

                phones = self._extract_phones(text)
                emails = self._extract_emails(text)

                logger.info(f"Impressum {url}: {len(phones)} phones, {len(emails)} emails")

                return ImpressumResult(
                    phones=phones,
                    emails=emails,
                    success=len(phones) > 0
                )

            except httpx.HTTPStatusError as e:
                logger.debug(f"Impressum page not found: {url} ({e.response.status_code})")
                return None
            except Exception as e:
                logger.debug(f"Impressum scrape failed: {url} - {e}")
                return None

    def _extract_phones(self, text: str) -> List[PhoneResult]:
        """Extract phone numbers from text."""
        phones = []
        seen = set()

        # German phone patterns - order matters, more specific first
        patterns = [
            # +49 format with full number
            r'\+49\s*\(?\d{1,4}\)?\s*[\d\s\-/\.]{6,}',
            # 0049 format
            r'0049\s*\(?\d{1,4}\)?\s*[\d\s\-/\.]{6,}',
            # Local format with area code (0xxx followed by number)
            r'0[1-9]\d{2,4}\s*[-/\s\.]*\d{2,}[\d\s\-/\.]*',
            # Labeled patterns - capture the full number after label
            r'(?:Tel(?:efon)?|Phone|Fon|Mobil|Handy|Telefax)[:\.\s]+\+?[\d\s\-/\.\(\)]{8,}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the number
                number = self._clean_phone_number(match)
                # Minimum 10 digits for a valid German phone (area + number)
                if number and number not in seen and len(number) >= 10:
                    seen.add(number)
                    phones.append(PhoneResult(
                        number=number,
                        type=self._determine_phone_type(number),
                        source=PhoneSource.IMPRESSUM
                    ))

        return phones

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        pattern = r'[\w\.\-+]+@[\w\.\-]+\.[a-zA-Z]{2,}'
        emails = re.findall(pattern, text)

        # Filter out common false positives
        filtered = []
        for email in emails:
            email_lower = email.lower()
            # Skip image files and common non-contact emails
            if any(x in email_lower for x in ['.png', '.jpg', '.gif', 'example.com']):
                continue
            filtered.append(email)

        return list(set(filtered))

    def _clean_phone_number(self, raw: str) -> str:
        """Clean and normalize phone number."""
        # Remove label text
        raw = re.sub(r'^.*?(?=[\d+])', '', raw)
        # Keep only digits and +
        cleaned = re.sub(r'[^\d+]', '', raw)

        # Normalize to +49 format
        if cleaned.startswith('+49'):
            # Remove extra 0 after +49 if present
            cleaned = re.sub(r'^\+490', '+49', cleaned)
        elif cleaned.startswith('0049'):
            # Convert 0049 to +49
            cleaned = '+49' + cleaned[4:].lstrip('0')
        elif cleaned.startswith('0'):
            # Convert leading 0 to +49
            cleaned = '+49' + cleaned[1:]
        elif len(cleaned) >= 10 and not cleaned.startswith('+'):
            # Assume German number without prefix, add +49
            cleaned = '+49' + cleaned

        return cleaned

    def _determine_phone_type(self, number: str) -> PhoneType:
        """Determine if phone is mobile or landline based on German prefixes."""
        # German mobile prefixes: 015x, 016x, 017x
        if re.match(r'(\+49|0049)?1[567]\d', number):
            return PhoneType.MOBILE

        # Austrian mobile: +43 6xx
        if re.match(r'(\+43|0043)?6\d', number):
            return PhoneType.MOBILE

        # Swiss mobile: +41 7x
        if re.match(r'(\+41|0041)?7[6789]\d', number):
            return PhoneType.MOBILE

        # Has country code or starts with 0 + area code = likely landline
        if re.match(r'(\+\d{2}|0\d{2,5})', number):
            return PhoneType.LANDLINE

        return PhoneType.UNKNOWN
