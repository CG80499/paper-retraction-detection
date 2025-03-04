import { PdfAnalyzer } from "@/components/PdfAnalyzer";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-6 bg-background">
      <PdfAnalyzer />
    </main>
  );
}
