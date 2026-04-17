## Dev environment tips
- Use uv for Python package management and manage this project as a uv project.
- There is a Python virtual environment in .venv; if for some reason you don't see it, run `scripts/setup_env.sh`.
- The `rg` command is installed; use it for quick searching.
- The git server we use is Github.
- Commit every change you make and ask the user to push changes when a significant batch of changes has been made.
- When you make a big discovery or change, note this in docs/LEARNINGS.md, even if user does not ask you to.
- Update README.md with each major commit.
- You can find the ComfyUI source code in ~/git/ComfyUI.

## Dev process tips
- Use Python logging liberally to insert judicious info, warning, and debug messages in the code.
- Import logging into each module and set logger = logging.getLogger(__name__). Then use logger for logging in that module.
- In most cases where an error message is called for, you should raise an appropriate exception. We want to know.
- Let exceptions bubble up to the caller unless there is a logical way to handle the exception within the current scope
.
- Unless there is no more specific alternative, never catch the base `Exception` class. Always catch the most specific
exception you can.
- Commit every change you make and ask the user to push changes when a significant batch of changes has been made.
- Use Python type hints, data classes, Pydantic models, and other strong typing features everywhere.
- Never write a function that has no typing on parameters.
- Write a documentation string for every function and every class.
- Keep functions short and modular. If a function grows longer than 50 rows or so, break it into separate functions tha
t you can compose in a sequence.
- Take the time to keep your code clean, tidy, and readable. Comments should not be required if code is written in a mo
dular, readable manner.

## Testing instructions
- Add or update tests for the code you change, even if nobody asked.
- Use pytest and create a harness so the user can just type "uv pytest".
- Always run the full test suite before every commit. Never commit if the test suite fails to pass.

