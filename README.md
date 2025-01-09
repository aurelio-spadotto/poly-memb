Workflow Organization:

- on branch main commit changes in the python library, and in docs/. Important: docs/ is only there to generate the documentation, commit changes only if you want to change the configuration fo the documentation, and always run a make clean to avoid committing files generated from doc compilation

- on branch gh-pages only the documentation in format html. The documentation is deployed by github-pages. Never commit directly to the branch.
When you have generated the doc, use
```ghp-import -n -p -f docs/_build/html
to update the branch
