# Usage Guide: OSINT Toolkit

## Running the Toolkit

- Activate your virtual environment
- Install requirements
- Run `python examples/run_osint.py`
- Enter a target value (username/domain/email)

## Modules Overview

- **SocialMediaOSINT**: Checks if a given username has a public Twitter profile.
- **DomainIntelligence**: Uses WHOIS to gather domain meta.
- **EmailFinder**: Verifies format compliance, not existence.
- **DNSLookup**: Gets A records for domains.
- **BreachDataChecker**: Uses HaveIBeenPwned (API key needed; store in `.env` as HIBP_API_KEY).
- **NewsScraper**: Gets top 5 HackerNews stories.

## API Keys & Environment Variables

- For HaveIBeenPwned, get key at [HPIB API Key page](https://haveibeenpwned.com/API/Key)
- Create `.env` file in project root:
HIBP_API_KEY=your_api_key_here

text

## Ethical Usage

- Respect terms of service for any public endpoint.
- Do NOT scrape private data or violate privacy.
- Use responsibly for legal research and educational purposes.

## Extending Toolkit

- Add more modules to `src/modules/`.
- Expand `IntelligenceGatherer` logic as needed.

---

*See README.md for main features and MIT License for usage terms.*
