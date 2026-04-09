# Notion API Configuration

## Access
- Parent page: Unlearning
- Parent URL: https://www.notion.so/Unlearning-337d00c73ea380848cc2c7f2956d5e15
- Parent ID: 337d00c7-3ea3-8084-8cc2-c7f2956d5e15

## Published Pages
- 2026-04-04: https://www.notion.so/2026-04-04-Progress-Report-Codex-OMC-337d00c73ea381d2a40fc8398f5f4541
- 2026-04-05: https://www.notion.so/2026-04-05-v20-v21-Multi-Concept-339d00c73ea3811a807dd73676daf983
- 2026-04-09: https://www.notion.so/33dd00c73ea3817bb002f9fdc1a08825

## Publishing Pattern
1. Prepare report in `docs/omc_reports/YYYY-MM-DD_*.md`
2. Create child page via API under parent page ID
3. Save receipt in `docs/omc_reports/YYYY-MM-DD_notion_publish_receipt.md`

## API Call Template (curl)
```bash
curl -s -X POST 'https://api.notion.com/v1/pages' \
  -H "Authorization: Bearer ${NOTION_TOKEN}" \
  -H "Content-Type: application/json" \
  -H "Notion-Version: 2022-06-28" \
  -d '{ "parent": {"page_id": "PARENT_ID"}, "properties": {...}, "children": [...] }'
```
