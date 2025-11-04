import whois
import validators

class DomainIntelligence:
    def lookup_domain(self, domain):
        if validators.domain(domain):
            try:
                w = whois.whois(domain)
                return {
                    "domain": domain,
                    "registrar": w.registrar,
                    "creation_date": w.creation_date,
                    "expiration_date": w.expiration_date
                }
            except Exception as e:
                return {"error": str(e)}
        return {"error": "Invalid domain"}
