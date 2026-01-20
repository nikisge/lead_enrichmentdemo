"""
Job URL Scraper - Extracts contact persons from original job posting URLs.

Hybrid approach:
1. First try httpx (fast, ~100ms) for simple sites
2. Fall back to Playwright for JS-heavy sites (LinkedIn, StepStone)

Goal: Find Ansprechpartner (contact person) with:
- Name (real person, not generic)
- Email (personal, not info@)
- Phone number
"""

import re
import logging
from typing import Optional, List
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Sites that need JavaScript rendering
JS_HEAVY_SITES = [
    'linkedin.com',
    'stepstone.de',
    'stepstone.at',
    'stepstone.ch',
    'xing.com',
]

# Sites where httpx should work fine
SIMPLE_SITES = [
    'indeed.com',
    'indeed.de',
    'monster.de',
    'arbeitsagentur.de',
    'meinestadt.de',
]


@dataclass
class ScrapedContact:
    """Contact person found on job posting page."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    title: Optional[str] = None
    source_url: str = ""
    confidence: float = 0.0  # 0-1, how confident we are this is a real contact


class JobUrlScraper:
    """Scrapes job posting URLs to find contact persons."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._playwright = None
        self._browser = None

    async def scrape_contact(self, url: str) -> Optional[ScrapedContact]:
        """
        Scrape job posting URL to find contact person.

        Returns ScrapedContact if found, None otherwise.
        """
        if not url:
            return None

        domain = self._get_domain(url)
        logger.info(f"Scraping job URL: {url} (domain: {domain})")

        html = None

        # Decide scraping method based on domain
        if self._needs_js_rendering(domain):
            logger.info(f"Using Playwright for JS-heavy site: {domain}")
            html = await self._scrape_with_playwright(url)

            # Fallback to httpx if Playwright fails
            if not html:
                logger.info("Playwright failed, falling back to httpx")
                html = await self._scrape_with_httpx(url)
        else:
            logger.info(f"Using httpx for simple site: {domain}")
            html = await self._scrape_with_httpx(url)

            # If httpx returns very little content, try Playwright as fallback
            if html and len(html) < 2000:
                logger.info("httpx returned minimal content, trying Playwright")
                playwright_html = await self._scrape_with_playwright(url)
                if playwright_html:
                    html = playwright_html

        if not html:
            logger.warning(f"Failed to scrape URL: {url}")
            return None

        # Extract contact from HTML
        contact = self._extract_contact(html, url)

        if contact and (contact.name or contact.email):
            logger.info(f"Found contact: {contact.name} / {contact.email}")
            return contact

        logger.info("No contact found in job posting")
        return None

    async def _scrape_with_httpx(self, url: str) -> Optional[str]:
        """Fast scraping with httpx (no JS rendering)."""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
                }
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.warning(f"httpx scraping failed: {e}")
            return None

    async def _scrape_with_playwright(self, url: str) -> Optional[str]:
        """JS-rendering with Playwright (slower but works for dynamic sites)."""
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        viewport={'width': 1280, 'height': 720},
                        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    )
                    page = await context.new_page()

                    # Navigate with timeout
                    await page.goto(url, wait_until='domcontentloaded', timeout=self.timeout * 1000)

                    # Wait a bit for dynamic content
                    await page.wait_for_timeout(1500)

                    # Get page content
                    html = await page.content()
                    return html

                finally:
                    await browser.close()

        except ImportError:
            logger.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")
            return None
        except Exception as e:
            logger.warning(f"Playwright scraping failed: {e}")
            return None

    def _extract_contact(self, html: str, source_url: str) -> Optional[ScrapedContact]:
        """Extract contact person from HTML."""
        soup = BeautifulSoup(html, 'lxml')

        # Remove script/style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        text = soup.get_text(separator='\n', strip=True)

        # Limit text size
        text = text[:20000]

        # Extract potential contacts
        name = None
        email = None
        phone = None
        title = None
        confidence = 0.0

        # 1. Find email addresses (prioritize personal emails)
        emails = self._extract_emails(text)
        personal_emails = [e for e in emails if not self._is_generic_email(e)]

        if personal_emails:
            email = personal_emails[0]
            # Try to extract name from email
            name_from_email = self._extract_name_from_email(email)
            if name_from_email:
                name = name_from_email
                confidence += 0.4

        # 2. Find contact section and extract name
        contact_name = self._find_contact_name(text)
        if contact_name:
            if not name:
                name = contact_name
            confidence += 0.3

        # 3. Find phone numbers near contact info
        phone = self._extract_phone_near_contact(text, name or email)
        if phone:
            confidence += 0.2

        # 4. Find job title/position
        title = self._extract_contact_title(text, name)
        if title:
            confidence += 0.1

        if not name and not email:
            return None

        return ScrapedContact(
            name=name,
            email=email,
            phone=phone,
            title=title,
            source_url=source_url,
            confidence=min(confidence, 1.0)
        )

    def _extract_emails(self, text: str) -> List[str]:
        """Extract all email addresses from text."""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(pattern, text.lower())))

    def _is_generic_email(self, email: str) -> bool:
        """Check if email is generic (not personal)."""
        generic_patterns = [
            'info@', 'kontakt@', 'contact@', 'office@', 'mail@',
            'bewerbung@', 'jobs@', 'karriere@', 'career@', 'hr@',
            'personal@', 'recruiting@', 'service@', 'support@',
            'hello@', 'team@', 'admin@', 'webmaster@', 'noreply@'
        ]
        return any(email.startswith(p) for p in generic_patterns)

    def _extract_name_from_email(self, email: str) -> Optional[str]:
        """Extract name from email like hans.mueller@company.de -> Hans Mueller."""
        local_part = email.split('@')[0]

        if '.' in local_part:
            parts = local_part.split('.')
            if len(parts) >= 2:
                first = parts[0]
                last = parts[-1]

                # Skip if too short (initials)
                if len(first) < 2 or len(last) < 2:
                    return None

                # Skip if contains numbers
                if any(c.isdigit() for c in first + last):
                    return None

                return f"{first.capitalize()} {last.capitalize()}"

        return None

    def _find_contact_name(self, text: str) -> Optional[str]:
        """Find contact person name in text using German patterns."""
        patterns = [
            # "Ihr Ansprechpartner: Max Müller"
            r'(?:Ihr\s+)?Ansprechpartner(?:in)?[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
            # "Kontakt: Max Müller"
            r'Kontakt[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
            # "Bewerbung an: Max Müller"
            r'Bewerbung(?:\s+an)?[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
            # "Fragen? Max Müller"
            r'Fragen\??[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
            # "Frau/Herr Max Müller"
            r'(?:Frau|Herr)\s+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if self._is_valid_name(name):
                    return name

        return None

    def _is_valid_name(self, name: str) -> bool:
        """Validate if text looks like a real person name."""
        if not name or len(name) < 5 or len(name) > 40:
            return False

        words = name.split()
        if len(words) < 2:
            return False

        # Filter out job titles
        invalid_patterns = [
            'präsident', 'teamleiter', 'leiter', 'manager', 'direktor',
            'geschäftsführ', 'ceo', 'cto', 'cfo', 'gmbh', 'ag',
            'abteilung', 'personal', 'recruiting', 'team'
        ]

        name_lower = name.lower()
        if any(p in name_lower for p in invalid_patterns):
            return False

        # Check if words look like names
        for word in words[:2]:
            if not word[0].isupper():
                return False
            clean = word.replace('-', '')
            if not clean.isalpha():
                return False

        return True

    def _extract_phone_near_contact(self, text: str, anchor: Optional[str]) -> Optional[str]:
        """Extract phone number near contact name/email."""
        # German phone patterns
        phone_pattern = r'(?:\+49|0049|0)\s*[\d\s\-/]{8,15}'

        phones = re.findall(phone_pattern, text)

        if phones:
            # Clean and return first valid phone
            for phone in phones:
                cleaned = re.sub(r'[^\d+]', '', phone)
                if len(cleaned) >= 10:
                    return phone.strip()

        return None

    def _extract_contact_title(self, text: str, name: Optional[str]) -> Optional[str]:
        """Extract job title near contact name."""
        if not name:
            return None

        # Find title near name
        name_pos = text.lower().find(name.lower())
        if name_pos == -1:
            return None

        # Look for title in surrounding text
        context = text[max(0, name_pos - 100):name_pos + len(name) + 100]

        title_patterns = [
            r'(Personalleiter(?:in)?)',
            r'(HR\s*Manager(?:in)?)',
            r'(Recruiter(?:in)?)',
            r'(Talent\s*Acquisition)',
            r'(Geschäftsführer(?:in)?)',
            r'(CEO|CTO|CFO|COO)',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return ""

    def _needs_js_rendering(self, domain: str) -> bool:
        """Check if domain needs JavaScript rendering."""
        return any(js_site in domain for js_site in JS_HEAVY_SITES)
