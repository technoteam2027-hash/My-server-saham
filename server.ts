import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured in environment.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
}

const LIA_SYSTEM_INSTRUCTION = `Tu es LiA (ليـا), l'assistante vocale intelligente officielle de Saham Bank (سهام بنك).
Tu parles couramment l'arabe (arabe standard et darija marocaine compréhensible) et le français.
Réponds TOUJOURS dans la même langue que la question de l'utilisateur (en arabe si l'utilisateur parle ou écrit en arabe/darija, en français s'il parle/écrit en français).

Tes domaines d'expertise principaux :
1. Diagnostic et résolution professionnelle des problèmes techniques sur smartphone et application mobile :
   - Problèmes de réseau, Wi-Fi, 4G/5G, lenteur de connexion ou erreurs de chargement.
   - Capteur d'empreinte digitale (Touch ID / biométrie) et reconnaissance faciale (Face ID) qui ne répondent pas.
   - Notifications push absentes, permissions micro/caméra bloquées, batterie/économie d'énergie bloquant l'arrière-plan.
   - Cache saturé, blocage ou fermeture soudaine de l'application, mise à jour iOS/Android, réception des codes SMS OTP.
   - Problèmes de carte sans contact (NFC), Apple Pay / Google Wallet ou affichage sur l'écran du téléphone.
2. Services Saham Bank : comptes chèques, comptes sur carnet, virements, cartes Visa, dotations internationales, RIB et sécurité.

Style de réponse :
- Professionnel, rassurant, clair et structuré pour être lu facilement sur mobile ET écouté à la voix.
- Donne des étapes pratiques courtes et précises (2 à 4 points concis maximum, sans blabla inutile).
- N'invente jamais de données bancaires privées non fournies dans le contexte.`;

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '12mb' }));

  // 1. LiA AI Chat Endpoint (Device Troubleshooting + Saham Bank Support)
  app.post('/api/lia/chat', async (req: Request, res: Response) => {
    try {
      const { message, history = [], language = 'fr' } = req.body as {
        message?: string;
        history?: ChatTurn[];
        language?: 'fr' | 'ar';
      };

      if (!message || typeof message !== 'string' || !message.trim()) {
        res.status(400).json({ error: 'Message is required.' });
        return;
      }

      const ai = getAiClient();

      const contents = [
        ...history.slice(-8).map((turn) => ({
          role: turn.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: turn.text }],
        })),
        {
          role: 'user',
          parts: [{ text: message.trim() }],
        },
      ];

      const requestConfig = {
        systemInstruction: `${LIA_SYSTEM_INSTRUCTION}\nLangue d'interface préférée de l'utilisateur : ${language === 'ar' ? 'Arabe' : 'Français'}.`,
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
      };

      let response;
      try {
        response = await ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents,
          config: requestConfig,
        });
      } catch (primaryErr: unknown) {
        const msg = primaryErr instanceof Error ? primaryErr.message : '';
        if (msg.includes('503') || msg.includes('UNAVAILABLE')) {
          response = await ai.models.generateContent({
            model: 'gemini-3.1-flash-lite',
            contents,
            config: {
              systemInstruction: requestConfig.systemInstruction,
            },
          });
        } else {
          throw primaryErr;
        }
      }

      const reply = (response.text ?? '').trim();
      if (!reply) {
        res.status(502).json({ error: 'Empty response from AI model.' });
        return;
      }

      res.json({ reply });
    } catch (error: unknown) {
      const errMsg = error instanceof Error ? error.message : 'Failed to generate AI response.';
      console.error('[LiA Chat Error]:', errMsg);
      res.status(500).json({ error: errMsg });
    }
  });

  // 2. LiA Voice Audio Transcription Endpoint (Phone Microphone Recording -> Text)
  app.post('/api/lia/transcribe', async (req: Request, res: Response) => {
    try {
      const { audioBase64, mimeType = 'audio/webm', language = 'fr' } = req.body as {
        audioBase64?: string;
        mimeType?: string;
        language?: 'fr' | 'ar';
      };

      if (!audioBase64 || typeof audioBase64 !== 'string') {
        res.status(400).json({ error: 'Audio data is required.' });
        return;
      }

      const ai = getAiClient();
      const promptHint =
        language === 'ar'
          ? 'استمع إلى التسجيل الصوتي واكتب نص الكلام بدقة تامة دون أي إضافات أو تعليقات. إذا كان الكلام بالعربية أو الدارجة أو الفرنسية اكتبه كما هو.'
          : 'Transcris exactement ce qui est dit dans cet audio (français ou arabe) sans ajouter de commentaires.';

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-transcribe',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: audioBase64,
              },
            },
            { text: promptHint },
          ],
        },
      });

      const text = (response.text ?? '').trim();
      res.json({ text });
    } catch (error: unknown) {
      const errMsg = error instanceof Error ? error.message : 'Audio transcription failed.';
      console.error('[LiA Transcribe Error]:', errMsg);
      res.status(500).json({ error: errMsg });
    }
  });

  // 3. LiA Text-to-Speech Voice Synthesis Endpoint (Optional Voice Output)
  app.post('/api/lia/tts', async (req: Request, res: Response) => {
    try {
      const { text } = req.body as { text?: string };
      if (!text || typeof text !== 'string' || !text.trim()) {
        res.status(400).json({ error: 'Text is required for TTS.' });
        return;
      }

      const ai = getAiClient();
      const cleanText = text.replace(/[*#_`]/g, '').slice(0, 700);

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash-lite-tts',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: cleanText,
                speechMetadata: {
                  style: 'Warm, calm, professional banking assistant voice',
                },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Kore' },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!base64Audio) {
        res.status(502).json({ error: 'No audio returned by TTS model.' });
        return;
      }

      res.json({ audioBase64: base64Audio, mimeType: 'audio/wav' });
    } catch (error: unknown) {
      const errMsg = error instanceof Error ? error.message : 'TTS generation failed.';
      console.error('[LiA TTS Error]:', errMsg);
      res.status(500).json({ error: errMsg });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Saham Bank server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});
