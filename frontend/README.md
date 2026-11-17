# Docs RAG — frontend

Web interface (Next.js, TypeScript, Tailwind CSS) to upload documents and chat with them through
the [backend API](../backend/README.md).

## Development

Requirements: Node.js 22+ and the backend running (by default on `http://localhost:8000`).

```bash
cd frontend
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL if the API is elsewhere
npm install
npm run dev                  # http://localhost:3000
```

| Script                 | Description                    |
| ---------------------- | ------------------------------ |
| `npm run dev`          | Development server             |
| `npm run build`        | Production build               |
| `npm run lint`         | ESLint                         |
| `npm run typecheck`    | TypeScript type-check          |
| `npm run format:check` | Check formatting with Prettier |

The header shows whether the API is reachable; it polls `GET /health` every 15 seconds.
