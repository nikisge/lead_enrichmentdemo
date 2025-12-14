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
