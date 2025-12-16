import logging
import re
import httpx
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)


class LinkedInSearchClient:
    """
    Search for LinkedIn profiles using Google Custom Search API.
    Useful when Apollo is not available (free plan limitation).
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.google_api_key
        self.cse_id = settings.google_cse_id
        self.timeout = settings.api_timeout

    async def find_linkedin_profile(
        self,
        name: str,
        company: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Optional[str]:
        """
        Search Google for a person's LinkedIn profile.

        Args:
            name: Full name of the person
            company: Company name (optional but improves accuracy)
            domain: Company domain (optional)

        Returns:
            LinkedIn profile URL or None
        """
        if not self.api_key or not self.cse_id:
            logger.warning("Google API key or CSE ID not configured")
            return None

        # Try multiple search strategies
        strategies = []

        # Strategy 1: Name + Company (most specific)
        if company:
            strategies.append(f'"{name}" "{company}" site:linkedin.com/in')

        # Strategy 2: Name + Company without quotes on company
        if company:
            strategies.append(f'"{name}" {company} site:linkedin.com/in')

        # Strategy 3: Name + Domain
        if domain:
            strategies.append(f'"{name}" {domain} site:linkedin.com/in')

        # Strategy 4: Just name (fallback)
        strategies.append(f'"{name}" site:linkedin.com/in')

        for query in strategies:
            result = await self._search_google(query, name, company)
            if result:
                return result

        logger.info("No LinkedIn profile found after all strategies")
        return None

    async def _search_google(
        self,
        query: str,
        name: str,
        company: Optional[str] = None
    ) -> Optional[str]:
        logger.info(f"LinkedIn search query: {query}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = "https://www.googleapis.com/customsearch/v1"

            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": 5  # Get top 5 results
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                total_results = data.get("searchInformation", {}).get("totalResults", "0")
                logger.info(f"Google returned {len(items)} results (total: {total_results})")

                if not items:
                    return None  # Try next strategy

                # Parse name for matching
                name_parts = name.lower().split()
                first_name = name_parts[0] if name_parts else ""
                last_name = name_parts[-1] if len(name_parts) > 1 else ""

                # Find best LinkedIn profile match
                best_match = None
                best_score = 0

                for item in items:
                    link = item.get("link", "")
                    title = item.get("title", "").lower()
                    snippet = item.get("snippet", "").lower()

                    # Must be a LinkedIn profile URL
                    if not self._is_linkedin_profile_url(link):
                        continue

                    # Calculate match score
                    score = 0

                    # Check name in title/snippet
                    if first_name and first_name in title:
                        score += 2
                    if last_name and last_name in title:
                        score += 3  # Last name more important
                    if first_name and first_name in snippet:
                        score += 1
                    if last_name and last_name in snippet:
                        score += 1

                    # Check company in snippet (if provided)
                    if company:
                        company_lower = company.lower()
                        company_words = [w for w in company_lower.split() if len(w) > 3]
                        for word in company_words:
                            if word in snippet or word in title:
                                score += 2
                                break

                    if score > best_score:
                        best_score = score
                        best_match = link

                # Require minimum score of 3 (at least last name match)
                if best_match and best_score >= 3:
                    logger.info(f"Found LinkedIn profile (score={best_score}): {best_match}")
                    return self._normalize_linkedin_url(best_match)

                return None  # Try next strategy

            except httpx.HTTPStatusError as e:
                logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"Google search failed: {e}")
                return None

    def _is_linkedin_profile_url(self, url: str) -> bool:
        """Check if URL is a LinkedIn profile (not company page)."""
        if not url:
            return False

        # Must contain linkedin.com/in/ for personal profiles
        return "linkedin.com/in/" in url.lower()

    def _normalize_linkedin_url(self, url: str) -> str:
        """Normalize LinkedIn URL to standard format."""
        # Remove query params and trailing slashes
        url = url.split("?")[0].rstrip("/")

        # Ensure https
        if url.startswith("http://"):
            url = url.replace("http://", "https://")

        # Remove locale prefixes like /de/
        url = re.sub(r'linkedin\.com/[a-z]{2}/in/', 'linkedin.com/in/', url)

        return url

    async def find_decision_maker(
        self,
        company: str,
        domain: Optional[str] = None,
        titles: Optional[list] = None,
        job_category: Optional[str] = None
    ) -> Optional[dict]:
        """
        Search for a decision maker at a company when no contact name is known.
        Uses Google to find LinkedIn profiles - OPTIMIZED: max 3 requests.

        Priority:
        1. HR / Recruiting / Personal (best for job postings)
        2. Department head matching job category
        3. General executives (fallback)

        Args:
            company: Company name
            domain: Company domain (optional)
            titles: List of titles to search for (optional)
            job_category: Job category like "IT", "Sales", etc.

        Returns:
            dict with 'name', 'title', 'linkedin_url' or None
        """
        if not self.api_key or not self.cse_id:
            logger.warning("Google API key or CSE ID not configured for decision maker search")
            return None

        # OPTIMIZED: Use combined searches (max 3 Google requests!)
        # Search 1: HR/Recruiting (combined query)
        hr_query = "Personalleiter OR HR OR Recruiting OR Personal"
        result = await self._search_decision_maker_combined(company, hr_query, domain)
        if result:
            return result

        # Search 2: Job-category specific (if category provided)
        if job_category:
            category_query = self._get_category_query(job_category)
            if category_query:
                result = await self._search_decision_maker_combined(company, category_query, domain)
                if result:
                    return result

        # Search 3: Executive fallback
        exec_query = "Geschäftsführer OR CEO OR Inhaber OR Managing Director"
        result = await self._search_decision_maker_combined(company, exec_query, domain)
        if result:
            return result

        logger.info(f"No decision maker found for {company} after 3 searches")
        return None

    def _get_category_query(self, category: str) -> Optional[str]:
        """Get combined search query for job category."""
        category_lower = category.lower()

        category_queries = {
            "it": "CTO OR IT-Leiter OR Head of IT OR Tech Lead",
            "software": "CTO OR Head of Engineering OR Tech Lead",
            "tech": "CTO OR IT-Leiter OR Head of IT",
            "sales": "Vertriebsleiter OR Sales Director OR Head of Sales",
            "vertrieb": "Vertriebsleiter OR Sales Director",
            "marketing": "CMO OR Marketing-Leiter OR Head of Marketing",
            "finance": "CFO OR Finanzleiter OR Head of Finance",
            "finanzen": "CFO OR Finanzleiter",
            "operations": "COO OR Betriebsleiter OR Operations",
            "produktion": "Produktionsleiter OR Werkleiter",
            "logistik": "Logistikleiter OR Supply Chain",
            "einkauf": "Einkaufsleiter OR Procurement",
            "consulting": "Partner OR Principal OR Director",
            "beratung": "Partner OR Principal OR Managing Consultant",
            "healthcare": "Chefarzt OR Medical Director OR Klinikleiter",
            "medizin": "Chefarzt OR Ärztlicher Direktor",
        }

        for key, query in category_queries.items():
            if key in category_lower:
                return query

        return None

    async def _search_decision_maker_combined(
        self,
        company: str,
        title_query: str,
        domain: Optional[str] = None
    ) -> Optional[dict]:
        """Search Google with combined title query (OR syntax)."""
        # Build search query with OR combinations
        query = f'({title_query}) "{company}" site:linkedin.com/in'

        logger.info(f"Decision maker search: {query[:80]}...")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = "https://www.googleapis.com/customsearch/v1"

            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": 10  # Get more results to find best match
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                if not items:
                    return None

                # Find best match from results
                for item in items:
                    link = item.get("link", "")
                    item_title = item.get("title", "")
                    snippet = item.get("snippet", "")

                    # Must be a LinkedIn profile
                    if not self._is_linkedin_profile_url(link):
                        continue

                    # Check if company name appears
                    company_lower = company.lower()
                    company_words = [w for w in company_lower.split() if len(w) > 2]

                    company_match = any(
                        word in item_title.lower() or word in snippet.lower()
                        for word in company_words
                    )

                    if not company_match:
                        continue

                    # Extract name from LinkedIn title
                    name = self._extract_name_from_linkedin_title(item_title)
                    if not name:
                        continue

                    linkedin_url = self._normalize_linkedin_url(link)
                    extracted_title = self._extract_title_from_snippet(snippet, "")

                    logger.info(f"Found: {name} ({extracted_title}) at {company}")
                    return {
                        "name": name,
                        "title": extracted_title,
                        "linkedin_url": linkedin_url
                    }

                return None

            except httpx.HTTPStatusError as e:
                logger.error(f"Google API error: {e.response.status_code}")
                return None
            except Exception as e:
                logger.error(f"Decision maker search failed: {e}")
                return None

    def _get_category_titles(self, category: str) -> list:
        """Get relevant department head titles based on job category."""
        category_lower = category.lower()

        category_map = {
            "it": ["IT-Leiter", "Head of IT", "CTO", "Tech Lead", "IT Director", "Head of Engineering"],
            "software": ["CTO", "Head of Engineering", "Tech Lead", "VP Engineering", "IT-Leiter"],
            "tech": ["CTO", "Head of IT", "Tech Lead", "IT-Leiter", "Head of Engineering"],
            "sales": ["Vertriebsleiter", "Sales Director", "Head of Sales", "VP Sales", "CSO"],
            "vertrieb": ["Vertriebsleiter", "Sales Director", "Head of Sales", "Verkaufsleiter"],
            "marketing": ["Marketing-Leiter", "Head of Marketing", "CMO", "Marketing Director"],
            "finance": ["CFO", "Finanzleiter", "Head of Finance", "Finance Director"],
            "finanzen": ["CFO", "Finanzleiter", "Head of Finance", "Kaufmännischer Leiter"],
            "operations": ["COO", "Operations Manager", "Betriebsleiter", "Head of Operations"],
            "produktion": ["Produktionsleiter", "Head of Production", "Werkleiter", "COO"],
            "logistik": ["Logistikleiter", "Head of Logistics", "Supply Chain Manager"],
            "einkauf": ["Einkaufsleiter", "Head of Procurement", "Purchasing Manager"],
            "personal": ["Personalleiter", "HR Manager", "Head of HR", "HR Director"],
            "hr": ["Personalleiter", "HR Manager", "Head of HR", "HR Director"],
            "consulting": ["Partner", "Managing Consultant", "Principal", "Director"],
            "beratung": ["Partner", "Managing Consultant", "Principal", "Geschäftsführer"],
            "healthcare": ["Klinikleiter", "Chefarzt", "Medical Director", "Geschäftsführer"],
            "medizin": ["Klinikleiter", "Chefarzt", "Medical Director", "Ärztlicher Direktor"],
        }

        # Find matching category
        for key, titles in category_map.items():
            if key in category_lower:
                return titles

        # Default: general management
        return ["Abteilungsleiter", "Team Lead", "Head of", "Manager"]

    async def _search_decision_maker_google(
        self,
        company: str,
        title: str,
        domain: Optional[str] = None
    ) -> Optional[dict]:
        """Search Google for a specific title at a company."""
        # Build search query
        query = f'"{title}" "{company}" site:linkedin.com/in'
        if domain:
            query = f'"{title}" "{company}" OR "{domain}" site:linkedin.com/in'

        logger.info(f"Decision maker search: {query}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = "https://www.googleapis.com/customsearch/v1"

            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": 5
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                if not items:
                    return None

                # Find the best match
                for item in items:
                    link = item.get("link", "")
                    item_title = item.get("title", "")
                    snippet = item.get("snippet", "")

                    # Must be a LinkedIn profile
                    if not self._is_linkedin_profile_url(link):
                        continue

                    # Check if company name appears in title or snippet
                    company_lower = company.lower()
                    company_words = [w for w in company_lower.split() if len(w) > 2]

                    company_match = any(
                        word in item_title.lower() or word in snippet.lower()
                        for word in company_words
                    )

                    if not company_match:
                        continue

                    # Extract name from LinkedIn title (usually "Name - Title | LinkedIn")
                    name = self._extract_name_from_linkedin_title(item_title)
                    if not name:
                        continue

                    linkedin_url = self._normalize_linkedin_url(link)

                    # Try to extract actual title from snippet
                    extracted_title = self._extract_title_from_snippet(snippet, title)

                    logger.info(f"Found decision maker: {name} ({extracted_title}) at {company}")
                    return {
                        "name": name,
                        "title": extracted_title,
                        "linkedin_url": linkedin_url
                    }

                return None

            except httpx.HTTPStatusError as e:
                logger.error(f"Google API error for decision maker search: {e.response.status_code}")
                return None
            except Exception as e:
                logger.error(f"Decision maker search failed: {e}")
                return None

    def _extract_name_from_linkedin_title(self, title: str) -> Optional[str]:
        """Extract person name from LinkedIn search result title."""
        # LinkedIn titles are usually "FirstName LastName - Title | LinkedIn"
        # or "FirstName LastName | LinkedIn"
        if not title:
            return None

        # Remove " | LinkedIn" or " - LinkedIn" suffix
        title = re.sub(r'\s*[\|\-]\s*LinkedIn.*$', '', title, flags=re.IGNORECASE)

        # Split by " - " or " – " to separate name from job title
        parts = re.split(r'\s*[\-–]\s*', title)
        if parts:
            name = parts[0].strip()
            # Basic validation: should have at least 2 words
            if len(name.split()) >= 2:
                return name

        return None

    def _extract_title_from_snippet(self, snippet: str, searched_title: str) -> str:
        """Try to extract actual job title from snippet, fallback to searched title."""
        # Common patterns in LinkedIn snippets
        patterns = [
            r'(Geschäftsführer(?:in)?)',
            r'(CEO)',
            r'(CTO)',
            r'(COO)',
            r'(CFO)',
            r'(Managing Director)',
            r'(Director\s+\w+)',
            r'(Head of\s+\w+)',
            r'(Leiter(?:in)?\s+\w+)',
            r'(VP\s+\w+)',
            r'(Founder)',
            r'(Inhaber(?:in)?)',
            r'(Owner)',
        ]

        for pattern in patterns:
            match = re.search(pattern, snippet, re.IGNORECASE)
            if match:
                return match.group(1)

        return searched_title


async def search_linkedin(
    name: str,
    company: Optional[str] = None,
    domain: Optional[str] = None
) -> Optional[str]:
    """
    Convenience function to search for LinkedIn profile.
    """
    client = LinkedInSearchClient()
    return await client.find_linkedin_profile(name, company, domain)


async def search_decision_maker(
    company: str,
    domain: Optional[str] = None,
    titles: Optional[list] = None
) -> Optional[dict]:
    """
    Search for a decision maker at a company when no contact name is known.
    Uses Google to find LinkedIn profiles of executives/managers.

    Returns:
        dict with 'name', 'title', 'linkedin_url' or None
    """
    client = LinkedInSearchClient()
    return await client.find_decision_maker(company, domain, titles)
