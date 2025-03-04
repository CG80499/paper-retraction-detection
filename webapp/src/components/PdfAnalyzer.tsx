"use client";

import React, { useState } from "react";
import { FileInput } from "@/components/ui/file-input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { AlertTriangleIcon, CheckCircleIcon, ThumbsUpIcon, BookOpenIcon } from "lucide-react";

interface AnalysisResult {
  retractedPercentage: number;
  analysis: string;
  chainOfThought: string;
}

export function PdfAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleFileSelect = (selectedFile: File) => {
    setError(null);
    setFile(selectedFile);
    setResult(null);
  };

  const analyzeFile = async () => {
    if (!file) {
      setError("Please select a PDF file first");
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      // Convert the file to base64 for the API request
      const base64Content = await readFileAsBase64(file);

      // Call the API route to analyze the PDF
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fileName: file.name,
          fileContent: base64Content,
        }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const readFileAsBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64String = reader.result as string;
        // Strip out the data URL prefix if it exists
        const base64Content = base64String.split(",")[1] || base64String;
        resolve(base64Content);
      };
      reader.onerror = () => reject(new Error("Failed to read file"));
      reader.readAsDataURL(file);
    });
  };

  return (
    <div className="max-w-3xl mx-auto p-6">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-2xl font-bold">CiteCop - Paper Retraction Detector</CardTitle>
        </CardHeader>
        <CardContent>
          <FileInput onFileSelect={handleFileSelect} />
          {file && (
            <div className="mt-4 p-3 bg-muted rounded-md flex items-center gap-2">
              <BookOpenIcon className="text-primary h-5 w-5" />
              <span className="text-sm font-medium">{file.name}</span>
            </div>
          )}
          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertTriangleIcon className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {isAnalyzing && (
            <div className="mt-6 space-y-2">
              <p className="text-sm text-muted-foreground text-center">
                Analyzing paper with Claude 3.7...
              </p>
              <Progress value={50} className="h-2" />
            </div>
          )}
        </CardContent>
        <CardFooter>
          <Button onClick={analyzeFile} disabled={!file || isAnalyzing} className="w-full">
            {isAnalyzing ? "Analyzing..." : "Check If Paper Should Be Retracted"}
          </Button>
        </CardFooter>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {result.retractedPercentage > 50 ? (
                <AlertTriangleIcon className="h-6 w-6 text-destructive" />
              ) : (
                <CheckCircleIcon className="h-6 w-6 text-green-500" />
              )}
              Analysis Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center mb-6">
              <div className="text-4xl font-bold mb-2">{result.retractedPercentage}%</div>
              <p className="text-muted-foreground">
                {result.retractedPercentage > 50
                  ? "Likelihood this paper should be retracted"
                  : "Confidence this paper is legitimate"}
              </p>
            </div>
            <div className="mt-4 p-4 bg-muted rounded-md">
              <p className="text-sm">{result.analysis}</p>
            </div>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" className="mt-6 w-full">
                  View Detailed Analysis Process
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Detailed Analysis Process</DialogTitle>
                </DialogHeader>
                <div className="whitespace-pre-wrap text-sm">{result.chainOfThought}</div>
              </DialogContent>
            </Dialog>
          </CardContent>
          <CardFooter>
            <Button variant="outline" className="w-full" onClick={() => setFile(null)}>
              <ThumbsUpIcon className="h-4 w-4 mr-2" />
              Check Another Paper
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
}
