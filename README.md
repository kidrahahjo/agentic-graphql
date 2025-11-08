# GraphQL Agent

Lightweight Strawberry GraphQL server that exposes a single `ask` query. The resolver inspects the incoming prompt and decides how to fulfill it by delegating to an MCP (Model Context Protocol) server.

## Features

## Getting Started

1. Install dependencies:

   ```bash
   uv sync  # or: pip install -e .[dev]
   ```

2. Export configuration (or place in a `.env`):

   ```bash
   cp env.sample .env  # includes a localhost alpha MCP
  ```

3. Start the GraphQL server:

   ```bash
   uvicorn graphql_agent.main:app --reload
   ```

4. Open `http://localhost:8000/graphql` in your browser and run:

   ```graphql
   {
     ask(prompt: "Give me all my alphas owned by me")
   }
   ```


## MCP Protocol

- Transport: **JSON-RPC 2.0** over HTTP POST.
- Rest of the docs TBA


## Development

- Install dev tools:

  ```bash
  uv sync --group dev
  ```

- Lint with Ruff:

  ```bash
  uv run ruff check .
  uv run ruff format .
  ```

- Type-check with MyPy:

  ```bash
  uv run mypy .
  ```