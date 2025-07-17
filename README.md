# Basic text-to-sql demo app

- This is a basic text-to-sql demo app with a streamlit frontend
- The fictional "Superstore sales" dataset is loaded into a SQLite DB and queried using natural language
- Uses Gemini 2.5 flash by default, and relies on `instructor-ai` for structured output, retries, etc.

## Guardrails

### Queries

- Always adds a limit of 1000 rows if not specified
- Only select-like statements are allowed (starting with a `WITH` clause for example is okay if it's a select), no DB modification queries are allowed