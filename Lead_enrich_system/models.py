from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class PhoneSource(str, Enum):
    KASPR = "kaspr"
    FULLENRICH = "fullenrich"
    IMPRESSUM = "impressum"
    COMPANY_MAIN = "company_main"


class PhoneType(str, Enum):
    MOBILE = "mobile"
    LANDLINE = "landline"
    UNKNOWN = "unknown"


# Webhook Input (from n8n)
class WebhookPayload(BaseModel):
    category: Optional[str] = None
    company: str
    date_posted: Optional[str] = None
    description: str
    id: str
    location: Optional[str] = None
    seen: Optional[bool] = False
    source: Optional[str] = None
    title: str
    url: Optional[str] = None


# LLM Parsing Result
class ParsedJobPosting(BaseModel):
    company_name: str
    company_domain: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    target_titles: List[str] = Field(default_factory=list)
    department: Optional[str] = None
    location: Optional[str] = None


# Decision Maker
class DecisionMaker(BaseModel):
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    apollo_id: Optional[str] = None


# Phone Result
class PhoneResult(BaseModel):
    number: str
    type: PhoneType = PhoneType.UNKNOWN
    source: PhoneSource
    formatted: Optional[str] = None


# Company Info
class CompanyInfo(BaseModel):
    name: str
    domain: Optional[str] = None
    industry: Optional[str] = None
    employee_count: Optional[str] = None
    location: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    linkedin_url: Optional[str] = None


# Company Intelligence for Sales
class CompanyIntel(BaseModel):
    """Company research data for sales preparation."""
    summary: str = ""  # AI-generated sales brief
    description: str = ""  # What the company does
    industry: str = ""
    employee_count: Optional[str] = None
    founded: Optional[str] = None
    headquarters: str = ""
    products_services: List[str] = Field(default_factory=list)
    hiring_signals: List[str] = Field(default_factory=list)
    website_url: str = ""


# Final Enrichment Result
class EnrichmentResult(BaseModel):
    success: bool
    company: CompanyInfo
    company_intel: Optional[CompanyIntel] = None  # Sales research data
    decision_maker: Optional[DecisionMaker] = None
    phone: Optional[PhoneResult] = None
    emails: List[str] = Field(default_factory=list)
    enrichment_path: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    # Original input reference
    job_id: str
    job_title: str
