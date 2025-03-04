"use client";

import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { UploadIcon } from "lucide-react";

interface FileInputProps {
  onFileSelect: (file: File) => void;
  className?: string;
  accept?: Record<string, string[]>;
  maxSize?: number;
  multiple?: boolean;
}

export function FileInput({
  onFileSelect,
  className,
  accept = { "application/pdf": [".pdf"] },
  maxSize = 10 * 1024 * 1024, // 10MB default
  multiple = false,
}: FileInputProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        if (multiple) {
          acceptedFiles.forEach(onFileSelect);
        } else {
          onFileSelect(acceptedFiles[0]);
        }
      }
    },
    [onFileSelect, multiple],
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors",
        isDragActive ? "border-primary bg-primary/10" : "border-input hover:border-primary/50",
        isDragReject && "border-destructive bg-destructive/10",
        className,
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center justify-center gap-2">
        <UploadIcon className="h-12 w-12 text-muted-foreground" />
        <div className="space-y-1">
          <p className="text-sm font-medium">
            {isDragActive ? "Drop the file here" : "Drag & drop or click to select"}
          </p>
          <p className="text-xs text-muted-foreground">
            PDF files only, up to {Math.round(maxSize / 1024 / 1024)}MB
          </p>
        </div>
        <Button variant="outline" size="sm" className="mt-2">
          Select File
        </Button>
      </div>
    </div>
  );
}
