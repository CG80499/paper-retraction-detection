# CiteCop

CiteCop is a tiny demonstration of using claude 3.7 to predict whether a paper should be
retracted based on the fulltext PDF of the paper.

# Design/Architecture

- NextJS app
- TailwindCSS
- Shadcn UI
- Lucide Icons
- Anthropic Claude 3.7 API
  - I will provide an API key to the app
- Prompt is hardcoded, placeholder for now
- Response will be parsed from XML

# Usage

- User will upload a PDF to the webapp
- We'll store that PDF statefully so they can come back to it later
- The link will be sharable
- The app will show the user the prediction as a big nicely formatted percentage
- The app will have an expandable section with the chain of thought process of the LLM
- The app will have a button to check another paper
