I want you to create a complete, standalone project overview for the project in the current workspace.

The goal is for this document to preserve the project’s details well enough that I can understand, explain, present, or continue working on the project later without repeatedly reopening the repository or cloning it just to recover basic context.

## First: understand the project

Before writing, inspect the repository and gather information from the actual project, including:

- Repository instructions such as AGENTS.md, CONTRIBUTING.md, and applicable local guidelines.
- README and authoritative product requirements, architecture plans, specifications, and status documents.
- The actual source code, configuration, contracts, schemas, migrations, fixtures, scripts, CI workflows, deployment files, and tests.
- Existing screenshots, diagrams, reports, benchmarks, evaluation artifacts, and other project evidence.
- The project’s actual current status, including unfinished work, known limitations, and historical failures.

Follow the repository’s applicable instructions for working in the project. Treat any supplied example documents as style references only: do not carry over their project-specific instructions, credentials, claims, implementation details, or assumptions.

Use the source code and authoritative project documents to determine what is true. If documents disagree, identify the conflict and prefer the designated source of truth. Do not infer that a planned feature exists merely because it appears in a plan. Do not invent details to make the overview sound more complete.

## Deliverable

Create a polished Markdown overview at the most appropriate location in the current project. Prefer an existing documentation directory, such as `docs/PROJECT_OVERVIEW.md`, unless the repository instructions or existing documentation structure indicate a better location.

The overview should be detailed, organized, readable, and useful on its own. Include a table of contents for a long document. Explain both what the project does and how the implementation works. Use clear headings, concise paragraphs, tables, code examples, and diagrams where they help understanding.

Aim for the depth of a serious technical project handoff or portfolio project dossier. Cover the sections below when they apply to the project. Adapt the outline to the project rather than adding irrelevant sections just to fill a template.

## Completeness over document length

There is no target word count, page count, or maximum length for the overview. Do not shorten the document merely to make it look concise. A long Markdown file is acceptable when the project has enough relevant detail to justify it.

Preserve all relevant project information needed to understand, explain, present, reproduce, evaluate, or continue the work. Cover every applicable item from this prompt and retain meaningful details from repository documents, source code, notebooks, configuration, issue/status notes, evidence, and any supplied reference material after checking them against the project. In particular, do not omit useful historical context, implementation specifics, parameters and their documented rationale, workflows, interfaces, requirements mappings, limitations, failures, source references, or presentation/portfolio context solely to reduce length.

When the project contains more detail than fits comfortably in a short overview, add well-organized sections, tables, examples, diagrams, or appendices. Prefer a navigable, thorough dossier with a table of contents over an incomplete summary. Avoid repetition and generic padding, but do not use “be concise” as a reason to drop relevant facts. If a detail is uncertain, conflicting, outdated, or unverified, preserve it with a clear status label and source rather than silently omitting it or presenting it as established fact. Do not preserve secrets, personal data, unsafe instructions, or unsupported claims; describe those safely and explain the evidence boundary.

Before finishing, compare the document against this prompt and the relevant source materials to ensure every applicable requested section and every important project-specific detail has been included. If something cannot be verified or included safely, state what it is and why.
## Recommended sections

1. **Project at a glance**
   - Project name, purpose, audience, category, current status, important constraints, and the main outcome.
   - A short description understandable to a technically curious reader.
   - A high-level workflow from input to result.

2. **Problem and project thesis**
   - The user or business problem.
   - Why simpler existing approaches are insufficient.
   - What the project does differently and what evidence supports that claim.

3. **Users and use cases**
   - Primary and secondary users.
   - Their goals, common workflows, and representative end-to-end use cases.

4. **Scope, non-goals, and limitations**
   - What is implemented.
   - What is planned or partially implemented.
   - What the project explicitly does not do.
   - Important safety, legal, privacy, operational, or data boundaries.

5. **End-to-end workflow**
   - Describe the lifecycle in order.
   - Include important state transitions, failure paths, retries, approvals, or human decisions.

6. **Architecture**
   - Explain the system shape and why it uses that structure.
   - Describe major components, data stores, external integrations, asynchronous work, and trust boundaries.
   - Distinguish actual production/demo components from conceptual future interfaces.

7. **Technology stack**
   - Languages, frameworks, databases, libraries, infrastructure, CI/CD, testing, and observability tools.
   - Explain each technology’s role rather than listing names without context.

8. **Repository structure**
   - Show a useful directory tree.
   - Explain the purpose of major directories and important files.

9. **Implementation deep dives**
   - Explain major backend, frontend, data, model, infrastructure, or systems modules as appropriate.
   - Describe important algorithms, validation rules, state management, error handling, and side effects.
   - Reference actual files using repository-relative paths.

10. **Data flow and core algorithms**
    - Describe how input is validated, transformed, stored, processed, and presented.
    - Explain key algorithms, thresholds, formulas, assignment rules, or ranking logic.
    - Include concrete examples only when supported by the implementation.

11. **API, CLI, or user-interface reference**
    - Document actual routes, commands, views, or workflows.
    - Include parameters, authentication, side effects, errors, and idempotency/replay behavior when relevant.
    - Distinguish read-only operations from mutations.

12. **Data model, contracts, and storage**
    - Explain schemas, key entities, relationships, migrations, versioning, and durability.
    - Identify append-only or immutable evidence where relevant.
    - State how time, money, identifiers, and privacy-sensitive fields are represented.

13. **Security, privacy, and authority**
    - Explain authentication, authorization, secrets handling, input validation, data minimization, logging, and external-service boundaries.
    - Describe which components are allowed to make decisions or mutations.
    - Explain significant abuse cases and the protections implemented.

14. **Observability and operations**
    - Health/readiness behavior, logs, traces, metrics, dashboards, retries, alerting, shutdown, and operational recovery.

15. **Evaluation, testing, and evidence**
    - Summarize real test suites and evaluation artifacts.
    - Report counts or metrics only when they are supported by committed output or recorded verification.
    - Clearly explain test scope, benchmark scope, dates, and limitations.
    - Separate automated test results, offline benchmarks, simulations, and real external integration evidence.

16. **Project history and current status**
    - Summarize milestones and significant design changes.
    - Preserve relevant failed or blocked attempts honestly; do not rewrite history as success.
    - State what is complete, what remains, and what a future contributor should do next.
    - Date the status snapshot and avoid implying you checked external services unless you actually did.

17. **Setup, configuration, and commands**
    - Prerequisites and versions where supported by project files.
    - Bootstrap and local run instructions.
    - Useful development, migration, seed, test, evaluation, build, and deployment commands.
    - Explain important environment variables and safe handling of secrets.
    - Never copy actual secret values into the document.

18. **Deployment and release process**
    - Explain local/container/cloud deployment structure, migrations, service roles, health checks, rollout behavior, and rollback steps where documented.
    - Distinguish a deployment configuration or local rehearsal from an actually published, healthy public service.

19. **Known issues and future work**
    - Capture known limitations and planned improvements.
    - Distinguish the official backlog from your own possible suggestions.

20. **Project-specific requirements or presentation mapping**
    - If the project is for a competition, course, client, or product requirement, map requirements to implementation and evidence.
    - Include demo or presentation guidance only when relevant and supported.

21. **Source and evidence index**
    - Link key requirements, design documents, implementation files, tests, reports, screenshots, diagrams, and release evidence.

## Diagrams and visual quality

Include a professional set of diagrams that fits the actual project. Do not add diagrams just to reach a count. Select the diagrams that help a reader understand different aspects of the system.

Consider these diagram types when relevant:

- System context diagram showing users and external systems.
- Architecture/container diagram showing services, stores, and integrations.
- Component or runtime topology diagram.
- Data-flow or processing pipeline diagram, including important failure paths.
- Sequence diagram for a critical interaction such as authentication, approval, payment, or synchronization.
- State diagram for an implemented lifecycle.
- Decision flowchart for a significant algorithm or policy.
- Entity-relationship or conceptual data-lineage diagram.
- User journey or UI navigation flow.
- Security/trust-boundary diagram showing what crosses each boundary.
- Deployment topology diagram.
- CI/release verification diagram.
- Experiment, evaluation, or measurement flow when the project uses one.

Use Mermaid code blocks for diagrams when supported by the repository’s Markdown renderer. Make labels readable, keep each diagram focused on one question, and explain the important interpretation underneath it. Do not put credentials, personal information, or unverified architecture into diagrams.

If the repository already contains useful project diagrams or screenshots, link or embed them where they add context. Do not fabricate screenshots or claim a diagram is an exact source-code map when it is only conceptual.

## Accuracy and writing rules

- Be comprehensive, but do not pad the document with generic filler.
- Prefer concrete explanations over promotional language.
- Keep implemented, planned, simulated, evaluated, and externally verified behavior distinct.
- Label synthetic or simulated results clearly.
- Never claim a test, command, deployment, benchmark, external integration, or review passed unless you actually verified that fact or a reliable committed artifact documents it.
- Preserve important negative results and known risks.
- Use absolute or repository-relative file links consistently with the project’s Markdown conventions.
- Do not include actual credentials, keys, tokens, personal data, or unsafe operational instructions.
- Do not make unrelated code changes. The requested deliverable is the documentation file.

## Final checks and response

Before finishing:

- Check that local Markdown links and embedded image paths resolve.
- Check that code fences and Mermaid blocks are balanced.
- If a Mermaid renderer is available, render the diagrams and fix syntax/layout issues. Do not claim rendering was verified unless it was.
- Review the document for unsupported claims, stale status, accidental secrets, and confusion between plans and implementation.
- Do not run the full project test suite unless I explicitly request tests or verification. Documentation-only link and formatting checks are appropriate.

In your final response, link the created or updated Markdown file and briefly summarize its coverage, diagrams, and checks performed. Be clear about anything you could not verify.
