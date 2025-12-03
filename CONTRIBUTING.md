# How to contribute to `Topologiq`

## Open issues
If you found a bug or would like a missing feature to be implemented, [create an issue](https://github.com/tqec/topologiq/issues/new/choose).

In order for your issue to be efficiently treated, please provide enough details as to be able to isolate or replicate the issue and include the appropriate labels.

## Asking a question
If you only want to ask a question, [create an issue](https://github.com/tqec/topologiq/issues/new/choose) using the "Ask a question" template. Provide as much information as possible.

## General overview of the process
Code contributions typically follow a standard process:

1. Check if there is an issue describing the problem you want to solve or the feature you want to implement in the [issues panel](https://github.com/tqec/topologiq/issues)
2. If there is no issue, [create an issue](https://github.com/tqec/topologiq/issues/new/choose) describing what you want to do
3. Ask for someone to assign you to the issue by commenting on the issue page
4. Fork the repository and create a new branch
5. Write down the code and submit a pull request when you think you fixed the issue or there is enough progress for others to look at your code
6. Wait for reviews and iterate with reviewers until the PR is satisfactory
7. Merge the PR and delete the branch; well done!

Please do not forget to open an issue and assign yourself **before** writing code, as this helps avoiding people working on the same feature/bug in parallel without knowing about each other.

And please always keep in mind we have all gotten PRs rejected, even good ones.

## Contributing guidelines
By contributing to this repository, you affirm your support for the following principles:
- You have the legal right to make the contribution.
- You are not currently a resident of a territory under sanctions or embargo by the United Nations (UN).
- Your submission does not knowingly violate export control laws or proprietary rights.
- You are not acting on behalf of any military or surveillance-related organization.
- You acknowledge and respect the ethical values expressed in Topologiq's [ETHICAL NOTICE](./ETHICAL_NOTICE.md).

## AI policy
Our Artificial Intelligence (AI) policy is that we want contributors who understand their own code and are capable of explaining it to a reviewer irrespective of whether they used AI or not.

Contributors who understand their code typically do one or several of the following (although rarely all, nor all at once):
- Discuss ideas in open Issues before submitting code
- Offer ideas of their own
- Disagree if they see better ways to achieve something
- Submit PRs that address one issue at a time
- Show actual thoughts in PR descriptions (as opposed to simply summarising technicalities)
- Give enough details in docstrings to understand things but are reasonable with length
- [If agenda and time zone allows] Assist and participate in weekly TQEC [meetings](https://meet.jit.si/TQEC-design-automation) (Wednesdays, 8:30am PST).

Additionally, please consider including an *AI disclaimer* in file- or function-level docstrings where usage of AI falls in either of the two categories below (include category and model but keep it short: examples [here](./src/topologiq/utils/grapher.py)):
- ***Coding partner***: AI generated >=60% of a script or function but was always under human supervision (sometimes also referred to as "co-pilot" mode)
- ***Automated AI***: AI generated the code in a completely or almost-completely autonomous capacity (sometimes also referred to as "Agentic AI" or "Vibe Coding").
