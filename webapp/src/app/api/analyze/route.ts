import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

// Initialize the Anthropic client
// We get the API key from environment variables for security
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || "",
});

export async function POST(request: Request) {
  try {
    // Check if API key is available
    if (!process.env.ANTHROPIC_API_KEY) {
      return NextResponse.json({ error: "Anthropic API key is not configured" }, { status: 500 });
    }

    // Parse the request body
    const { fileName, fileContent } = await request.json();

    if (!fileName || !fileContent) {
      return NextResponse.json({ error: "File name and content are required" }, { status: 400 });
    }

    // The prompt for Claude
    const prompt = `
      You are CiteCop, an AI assistant specializing in identifying academic papers that should be retracted.
      
      Please analyze the following academic paper and determine if it should be retracted.
      
      For your analysis, consider factors such as:
      1. Evidence of data fabrication or falsification
      2. Plagiarism or self-plagiarism
      3. Methodological errors that invalidate the results
      4. Ethical violations in research conduct
      5. Conflicts of interest that weren't disclosed
      6. Image manipulation or duplication
      7. Statistical errors or p-hacking
      
      Format your output as XML with the following structure:
      <result>
        <percentage>A number from 0-100 representing the likelihood this paper should be retracted</percentage>
        <summary>A concise summary of your analysis (2-3 sentences)</summary>
        <explanation>A detailed explanation of your reasoning</explanation>
      </result>
    `;

    // Make the API request to Claude
    const message = await anthropic.messages.create({
      model: "claude-3-5-sonnet-20240620", // Using Sonnet to save costs; adjust based on needs
      max_tokens: 4000,
      system:
        "You are CiteCop, an AI assistant specializing in identifying papers that should be retracted.",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: prompt,
            },
            {
              type: "document",
              source: {
                type: "base64",
                media_type: "application/pdf",
                data: fileContent,
              },
            },
          ],
        },
      ],
    });

    // Parse the XML response from Claude
    const responseText =
      typeof message.content[0] === "object" && "text" in message.content[0]
        ? (message.content[0].text as string)
        : "";

    // Extract data using indexOf and substring for better compatibility
    const getTagContent = (text: string, tag: string) => {
      const startTag = `<${tag}>`;
      const endTag = `</${tag}>`;
      const startIndex = text.indexOf(startTag) + startTag.length;
      const endIndex = text.indexOf(endTag);
      if (startIndex === -1 || endIndex === -1) return null;
      return text.substring(startIndex, endIndex);
    };

    const percentageStr = getTagContent(responseText, "percentage");
    const summary = getTagContent(responseText, "summary");
    const explanation = getTagContent(responseText, "explanation");

    if (!percentageStr || !summary || !explanation) {
      throw new Error("Failed to parse Claude response");
    }

    const retractedPercentage = parseInt(percentageStr, 10);

    // Return the parsed analysis
    return NextResponse.json({
      retractedPercentage,
      analysis: summary,
      chainOfThought: explanation,
      fileName,
    });
  } catch (error) {
    console.error("Error analyzing PDF:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "An unknown error occurred" },
      { status: 500 },
    );
  }
}
