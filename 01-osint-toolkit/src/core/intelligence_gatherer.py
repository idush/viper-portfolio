from src.modules.social_media_osint import SocialMediaOSINT
from src.modules.domain_intelligence import DomainIntelligence
from src.modules.email_finder import EmailFinder
from src.modules.data_correlation import DataCorrelation
from src.modules.dns_lookup import DNSLookup
from src.modules.breach_data_checker import BreachDataChecker
from src.modules.news_scraper import NewsScraper

class IntelligenceGatherer:
    def __init__(self):
        self.social = SocialMediaOSINT()
        self.domain = DomainIntelligence()
        self.email = EmailFinder()
        self.correlation = DataCorrelation()
        self.dns = DNSLookup()
        self.breach = BreachDataChecker()
        self.news = NewsScraper()

    def search_target(self, target):
        result = {
            "social": self.social.search_username(target),
            "domain": self.domain.lookup_domain(target),
            "email_valid": self.email.is_valid_email(target),
            "dns_records": self.dns.get_records(target),
            "breach_data": self.breach.check_email(target),
            "latest_news": self.news.get_latest_hackernews(),
        }
        result["correlation"] = self.correlation.correlate(result["social"], result["domain"])
        return result
