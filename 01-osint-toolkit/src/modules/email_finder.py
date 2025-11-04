import validators

class EmailFinder:
    def is_valid_email(self, email):
        return validators.email(email)
