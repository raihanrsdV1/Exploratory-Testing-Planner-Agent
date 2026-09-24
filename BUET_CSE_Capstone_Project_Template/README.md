# CSE 450 Capstone Project — Report Template

LaTeX template for the CSE 450 capstone project report, Department of CSE, BUET.
The chapters and sections are arranged so that every assessment indicator of the
CSE 450 rubric has a specific place in the report; the mapping is given below.

## Build

Overleaf: upload this folder as a zip, set the compiler to pdfLaTeX, and compile
`template.tex`.

Local (TeX Live / MacTeX / MiKTeX):

```
latexmk -pdf template.tex
```

## How to use

1. Edit the block at the top of `template.tex` (title, session, submission date,
   supervisor) and the team member names in `frontmatter/titlepage.tex` and
   `frontmatter/declaration.tex`.
2. Write each chapter in its file under `chapters/`. Every section carries a
   short gray italic `\guidance{...}` note saying what is expected there.
   Replace each note with your content.
3. The sections that are already present must all be covered as their notes
   describe, since each one corresponds to an assessment indicator. Beyond
   that, add chapters, sections, figures, and tables as you see fit to present
   your work best. The skeleton is a minimum, not a limit.
4. Put images in `figures/` and replace `\figplaceholder{...}` with
   `\includegraphics[width=...]{file}`.
5. Cite all sources. Add a bibliography (for example `\bibliographystyle{IEEEtran}`
   and `\bibliography{references}` before `\end{document}` in `template.tex`, with
   entries in `references.bib`).

## Files

| File                             | Content                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `template.tex`                   | Project details and chapter list                                             |
| `cse-capstone.sty`               | Formatting; no need to edit                                                  |
| `frontmatter/`                   | Title page, declaration, acknowledgement, abstract                           |
| `chapters/problem.tex`           | Background, problem statement, complexity, objectives and scope              |
| `chapters/research.tex`          | Literature review, existing technologies, market research, feasibility study |
| `chapters/requirements.tex`      | Stakeholders, requirements and constraints, standards, alternative solutions |
| `chapters/design.tex`            | Architecture, detailed design, tools, implementation, deployment             |
| `chapters/evaluation.tex`        | Experimental investigation, evaluation against requirements, design revisions|
| `chapters/management_impact.tex` | Planning, budget, economics, teamwork, society, sustainability, ethics       |
| `chapters/conclusion.tex`        | Conclusion, future work, self-directed learning                              |
| `figures/`                       | Images                                                                       |

## Rubric coverage

| PO   | Assessment indicator                               | Section                                                                                    |
| ---- | -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| PO1  | 1.1 Depth of Knowledge                             | 1.1 Background and Motivation; 1.2 Problem Statement; 1.3 Complexity of the Problem; 2.1 Literature Review; 2.2 Existing Solutions and Technologies |
| PO2  | 2.1 Formulation of Solutions                       | 1.2 Problem Statement (problem formulation); 2.1 Literature Review; 3.3 Formulation of Solutions |
| PO3  | 3.1 Regulatory Requirements and Standards          | 3.2 Regulatory Requirements and Standards                                                  |
| PO3  | 3.2 Project Objectives and Constraints             | 1.4 Project Objectives and Scope; 3.1 Project Objectives and Constraints                   |
| PO3  | 3.3 Functional Design and Sub-problem Partitioning | 4.1 Functional Design and Sub-problem Partitioning                                         |
| PO3  | 3.4 Design Refinement and Simulation               | 4.2 Design Refinement and Simulation; 5.3 Deviations and Design Revisions                  |
| PO4  | 4.1 Performance Evaluation                         | 5.1 Experimental Investigation; 5.2 Performance Evaluation; 5.3 Deviations and Design Revisions |
| PO5  | 5.1 Application of Tools                           | 4.3 Application of Tools                                                                   |
| PO6  | 6.1 Impact on Society                              | 6.3 Impact on Society                                                                      |
| PO7  | 7.1 Life Cycle Sustainability                      | 4.5 Deployment; 6.4 Environmental Impact and Life Cycle Sustainability                     |
| PO8  | 8.1 Equity and Inclusivity                         | 6.5.1 Equity and Inclusivity                                                               |
| PO8  | 8.2 Accountability and Personal Responsibility     | 6.5.2 Accountability and Personal Responsibility; Candidates' Declaration                  |
| PO8  | 8.3 Proper Use of Intellectual Property            | 6.5.3 Proper Use of Intellectual Property; References                                      |
| PO8  | 8.4 Professionalism and Ethical Codes              | 6.5.4 Professionalism and Ethical Codes                                                    |
| PO9  | 9.1 Participation and Contribution                 | 6.2.1 Participation and Contribution                                                       |
| PO9  | 9.2 Collaboration and Conflict Resolution          | 6.2.2 Collaboration and Conflict Resolution                                                |
| PO9  | 9.3 Leadership and Direction                       | 6.2.3 Leadership and Direction                                                             |
| PO9  | 9.4 Multidisciplinary Engagement                   | 6.2.4 Multidisciplinary Engagement                                                         |
| PO10 | 10.1 Written Ideas and Clarity                     | Whole report                                                                               |
| PO10 | 10.2 Visual Integration in Documentation           | Figures and tables throughout                                                              |
| PO10 | 10.3–10.5 Oral Delivery, Visual Aids, Q&A          | Assessed at the presentation, not in the report                                            |
| PO11 | Planning and Risk Management                       | 6.1.1 Planning and Risk Management                                                         |
| PO11 | Budgeting and Resource Identification              | 6.1.2 Budgeting and Resource Identification                                                |
| PO11 | Economic Analysis and Financial Prospecting        | 2.3 Market and User Research; 2.4 Feasibility Study; 6.1.3 Economic Analysis and Financial Prospecting |
| PO11 | Sustainability in Business and Commercialisation   | 6.1.4 Sustainability in Business and Commercialisation                                     |
| PO12 | 12.1 Continuous Engagement                         | 7.2 Continuous Engagement and Self-directed Learning                                        |

Section numbers refer to the template as supplied; if you insert chapters or
sections, the numbers shift but the headings stay the same.
