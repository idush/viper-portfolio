import dns.resolver

class DNSLookup:
    def get_records(self, domain):
        try:
            records = dns.resolver.resolve(domain, 'A')
            return [r.to_text() for r in records]
        except Exception as e:
            return {"error": str(e)}
