# CaseCraft in Layman's Terms

This is the simple version of the project overview.

- For the detailed technical version, see [README.md](../README.md).
- For a concise technical stakeholder summary, see [CASECRAFT_EXECUTIVE.md](./CASECRAFT_EXECUTIVE.md).

## What Is CaseCraft?

CaseCraft is a tool that reads a feature document and turns it into test cases.

Think of it like this:

- You give it a product requirement document.
- It reads what the feature is supposed to do.
- It looks at related product knowledge if you have provided any.
- It writes a structured list of checks that a tester or QA engineer can run.

In simple terms, it is a smart assistant for software testing.

## Why Does It Exist?

Writing test cases by hand takes time.

People often miss:

- edge cases
- negative cases
- cross-feature impacts
- regression risks
- consistency in formatting

CaseCraft helps reduce that manual effort. It does not replace human testers, but it helps them start faster and cover more ground.

## Explain It Like I'm Five

Imagine you are building a toy vending machine.

You write instructions saying:

- when someone puts in a coin, the toy should drop
- if there is no toy left, show a message
- if the wrong coin is used, reject it

Normally, a tester would read those instructions and make a checklist:

- Does the toy drop with the right coin?
- Does it reject the wrong coin?
- Does it show an empty message when out of toys?

CaseCraft does that checklist-writing job automatically.

It reads the instructions, thinks about what could go right or wrong, and produces a clean set of test cases.

## What Goes In?

CaseCraft can read:

- PDF files
- text files
- markdown files

These files usually contain feature requirements, business rules, workflows, or supporting documentation.

You can also feed it product knowledge such as:

- internal docs
- web pages
- rules and process documents
- previously collected product information

That extra knowledge helps it generate smarter and more grounded test cases.

## What Comes Out?

CaseCraft produces structured test suites in:

- Excel format
- JSON format

Each test case can include:

- use case
- test case title
- test type
- preconditions
- test data
- test steps
- expected results
- priority
- dependencies
- tags

So instead of getting a messy paragraph, you get something that is ready for QA use.

## How Does It Work?

At a high level, CaseCraft does this:

1. It reads your feature file.
2. It breaks the file into smaller parts so the AI can process it safely.
3. It looks up related product knowledge from its knowledge base.
4. It sends the feature content plus the related knowledge to an AI model.
5. It gets back draft test cases.
6. It cleans the output, fixes structure problems, removes duplicates, and organizes the result.
7. It saves the final test suite.

So the pipeline is basically:

read -> understand -> enrich -> generate -> clean -> export

## What Makes It Better Than a Simple Prompt?

A normal prompt might say:

"Generate test cases for this feature."

That can work, but it often produces generic answers.

CaseCraft goes further because it:

- uses your actual product documents
- searches a knowledge base for related context
- supports structured output
- validates and cleans AI responses
- reduces duplicate test cases
- can add reviewer-style polishing

That means the output is usually more useful than a one-off chat response.

## What Is the Knowledge Base in Simple Words?

The knowledge base is like a memory shelf.

You put useful project information on that shelf first. Later, when CaseCraft is generating test cases, it pulls the most relevant pieces off the shelf and uses them as context.

This helps it answer questions like:

- What rules already exist?
- What related feature might be affected?
- What business logic should not be broken?

Without that memory shelf, the AI has to guess more.

## Why Does It Use AI?

Because software requirements are written in human language, not code.

AI is good at reading language, spotting intent, and turning that intent into structured checks.

CaseCraft uses AI to:

- understand feature descriptions
- infer test scenarios
- identify edge cases
- generate readable test steps
- improve the quality of the final test suite

## What Parts of the Project Matter Most?

Here is the project in simple words:

- `cli/` is how you run the tool.
- `core/parser.py` reads files.
- `core/generator.py` runs the main workflow.
- `core/llm_client.py` talks to the AI model.
- `core/knowledge/` stores and retrieves product knowledge.
- `core/exporter.py` writes the final result to Excel or JSON.

If you only remember one thing, remember this:

CaseCraft takes human-written requirements and turns them into QA-ready test cases with AI plus product knowledge.

## What It Is Good At

CaseCraft is useful when you want to:

- save time writing test cases
- improve consistency in QA documentation
- include product context in generated tests
- cover negative and edge scenarios more reliably
- produce structured outputs for teams to review quickly

## What It Does Not Replace

CaseCraft is helpful, but it is not a full replacement for:

- QA judgement
- exploratory testing
- domain expertise
- bug triage
- final review of critical test cases

It is best used as a strong first draft generator and knowledge-assisted QA helper.

## A Simple Real-World Example

If a product team writes:

"Users can reset their password using email OTP. OTP expires in 5 minutes. Accounts lock after 5 failed attempts."

CaseCraft can help produce tests like:

- reset works with valid OTP
- expired OTP is rejected
- wrong OTP is rejected
- account locks after repeated failed attempts
- locked account behavior is shown correctly
- resend OTP flow works as expected

If the knowledge base already contains related login or security rules, CaseCraft can use that too.

## Where the Project Is Going Next

The next phase is to make CaseCraft smarter from real project history.

Two important directions are:

1. Fine-tuning models on high-quality test case examples so output becomes more accurate and more consistent.
2. Using already reported bugs as learning context so the system can generate tests around real human-discovered edge cases and regressions.

That means the tool will not just generate tests from requirements. It will also learn from what has already gone wrong in the real product.

## One-Sentence Summary

CaseCraft is an AI-powered assistant that reads feature documents, uses project knowledge, and produces structured test cases so QA teams can move faster and miss less.