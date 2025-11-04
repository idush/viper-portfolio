from src.core.intelligence_gatherer import IntelligenceGatherer

def main():
    target = input("Enter username, domain, or email to investigate: ").strip()
    gatherer = IntelligenceGatherer()
    result = gatherer.search_target(target)
    print("--- OSINT Results ---")
    for k, v in result.items():
        print(f"{k}:\n{v}\n")

if __name__ == "__main__":
    main()
