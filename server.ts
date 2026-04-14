import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import cors from "cors";
import fs from "fs";
import * as git from "isomorphic-git";
import http from "isomorphic-git/http/node";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(cors());
  app.use(express.json());

  // API Routes
  app.post("/api/ingest", async (req, res) => {
    const { repoUrl } = req.body;
    if (!repoUrl) return res.status(400).json({ error: "Repo URL is required" });

    try {
      const dir = path.join(process.cwd(), "temp-repo");
      if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true });
      }
      fs.mkdirSync(dir);

      console.log(`Cloning ${repoUrl}...`);
      await git.clone({
        fs,
        http,
        dir,
        url: repoUrl,
        singleBranch: true,
        depth: 1,
      });

      const files = await getAllFiles(dir);
      const chunks = [];

      for (const file of files) {
        const content = fs.readFileSync(file, "utf-8");
        const relativePath = path.relative(dir, file);
        
        // Smart chunking: split by functions/classes or blocks
        const blocks = content.split(/\n(?=(?:export|class|function|interface|const|let|var|import)\s)/);
        for (const block of blocks) {
          if (block.trim().length > 50) {
            chunks.push({
              text: block,
              metadata: { path: relativePath }
            });
          }
        }
      }

      res.json({ message: "Cloning and chunking complete", chunks });
    } catch (error: any) {
      console.error("Ingestion error:", error);
      res.status(500).json({ error: error.message });
    }
  });

  async function getAllFiles(dir: string): Promise<string[]> {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    const files = entries
      .filter(file => !file.isDirectory() && !file.name.startsWith(".") && !file.name.endsWith(".png") && !file.name.endsWith(".jpg"))
      .map(file => path.join(dir, file.name));
    const folders = entries.filter(folder => folder.isDirectory() && !folder.name.startsWith(".") && folder.name !== "node_modules" && folder.name !== ".git");
    for (const folder of folders) {
      files.push(...(await getAllFiles(path.join(dir, folder.name))));
    }
    return files;
  }

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
