
import { GoogleGenAI, HarmBlockThreshold, HarmCategory } from "@google/genai";
import { APP_CONFIG } from "../constants";

const SAFETY_SETTINGS = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
];

async function optimizeImage(base64: string, maxWidth = 512): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = base64;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      let width = img.width;
      let height = img.height;
      if (width > height) {
        if (width > maxWidth) { height *= maxWidth / width; width = maxWidth; }
      } else {
        if (height > maxWidth) { width *= maxWidth / height; height = maxWidth; }
      }
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return reject(new Error("Canvas failure"));
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
      resolve(canvas.toDataURL('image/jpeg', 0.8));
    };
    img.onerror = () => reject(new Error("Bildverarbeitung fehlgeschlagen."));
  });
}

function getCleanBase64(dataUrl: string): string {
  return dataUrl.replace(/^data:[^;]+;base64,/, "");
}

function getAI() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === "undefined") {
    throw new Error("INVALID_KEY");
  }
  return new GoogleGenAI({ apiKey });
}

export async function performVirtualTryOn(userBase64: string, productBase64: string, productName: string): Promise<{ image: string, size: string }> {
  const [optUser, optProduct] = await Promise.all([
    optimizeImage(userBase64, 512),
    optimizeImage(productBase64, 512)
  ]);

  const ai = getAI();
  
  const makeRequest = async () => {
    return await ai.models.generateContent({
      model: APP_CONFIG.IMAGE_MODEL,
      contents: {
        parts: [
          { text: `VIRTUAL TRY-ON:
          - Dress person in IMAGE 1 with outfit from IMAGE 2 (${productName}).
          - Keep person's face, hair, and pose identical.
          - Replace sleeves to match IMAGE 2.
          - Suggest best size (XS, S, M, L, XL, XXL).
          - Output: Image + Size (e.g. "Size: M").` },
          { inlineData: { data: getCleanBase64(optUser), mimeType: "image/jpeg" } },
          { inlineData: { data: getCleanBase64(optProduct), mimeType: "image/jpeg" } },
        ],
      },
      config: { 
        imageConfig: { 
          aspectRatio: "3:4",
          imageSize: "512px"
        },
        safetySettings: SAFETY_SETTINGS
      }
    });
  };

  try {
    let response;
    let retries = 0;
    const maxRetries = 2;
    
    while (retries <= maxRetries) {
      try {
        response = await makeRequest();
        break;
      } catch (err: any) {
        if ((err.message?.includes("429") || err.message?.includes("RESOURCE_EXHAUSTED")) && retries < maxRetries) {
          retries++;
          await new Promise(resolve => setTimeout(resolve, retries * 3000)); // Wait 3s, then 6s
          continue;
        }
        throw err;
      }
    }

    if (!response) throw new Error("Keine Antwort von der KI.");

    const candidates = response.candidates;
    if (!candidates || candidates.length === 0) {
      throw new Error("Die KI hat keine Antwort geliefert. Bitte versuche es mit einem anderen Foto erneut.");
    }

    const firstCandidate = candidates[0];
    
    if (firstCandidate.finishReason === 'SAFETY') {
      throw new Error("SAFETY_BLOCK");
    }

    const content = firstCandidate.content;
    if (!content || !content.parts || content.parts.length === 0) {
      throw new Error("Die KI-Antwort war leer. Bitte versuche es mit einem anderen Foto erneut.");
    }

    const parts = content.parts;
    
    let generatedImage = "";
    let recommendedSize = "M";

    // Extract image and size from parts
    for (const part of parts) {
      if (part.inlineData?.data) {
        generatedImage = `data:image/jpeg;base64,${part.inlineData.data}`;
      } else if (part.text) {
        const text = part.text.toUpperCase();
        const validSizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL'];
        for (const s of validSizes) {
          if (text.includes(s)) {
            recommendedSize = s;
            break;
          }
        }
        
        // Check for policy/error text
        const lowerText = part.text.toLowerCase();
        if (lowerText.includes("sorry") || lowerText.includes("cannot") || lowerText.includes("unable") || lowerText.includes("policy")) {
          throw new Error("Die KI konnte dieses Bild leider nicht verarbeiten. Bitte versuche ein Foto mit neutralerem Hintergrund.");
        }
      }
    }

    if (!generatedImage) {
      throw new Error("Die KI hat kein Bild generiert. Bitte versuche ein anderes Foto oder eine andere Pose.");
    }

    return { image: generatedImage, size: recommendedSize };
  } catch (err: any) {
    if (err.message?.includes("429") || err.message?.includes("RESOURCE_EXHAUSTED")) {
      throw new Error("Das API-Limit wurde erreicht. Bitte versuche es in ein paar Minuten erneut.");
    }
    if (err.message === "SAFETY_BLOCK") {
      throw new Error("Das Bild wurde aus Sicherheitsgründen abgelehnt. Bitte versuche ein Foto mit neutralerem Hintergrund oder einer anderen Pose.");
    }
    throw err;
  }
}

export async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
  });
}

export async function urlToBase64(url: string): Promise<string> {
  if (url.startsWith('data:')) return url;
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width; canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (ctx) { ctx.drawImage(img, 0, 0); resolve(canvas.toDataURL('image/jpeg', 0.9)); }
    };
    img.onerror = () => reject(new Error("Ladefehler"));
    img.src = `https://images.weserv.nl/?url=${encodeURIComponent(url)}&w=1024&output=jpg`;
  });
}
