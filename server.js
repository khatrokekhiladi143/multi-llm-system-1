const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;

const cache = new Map();

function getCacheKey(msg) {
  return require('crypto').createHash('md5').update(JSON.stringify(msg)).digest('hex');
}

async function callGroq(messages, apiKey) {
  const cacheKey = `groq-${getCacheKey(messages)}`;
  if (cache.has(cacheKey)) return cache.get(cacheKey);

  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      messages,
      max_tokens: 2000,
      temperature: 0.7,
    }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error?.message || 'Groq error');

  const result = {
    provider: 'groq',
    content: data.choices[0].message.content,
    tokens: data.usage.total_tokens,
  };

  cache.set(cacheKey, result);
  return result;
}

async function callGemini(messages, apiKey) {
  const cacheKey = `gemini-${getCacheKey(messages)}`;
  if (cache.has(cacheKey)) return cache.get(cacheKey);

  const contents = messages.map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }));

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents,
        generationConfig: { maxOutputTokens: 2000, temperature: 0.7 },
      }),
    }
  );

  const data = await res.json();
  if (!res.ok) throw new Error(data.error?.message || 'Gemini error');

  const result = {
    provider: 'gemini',
    content: data.candidates[0].content.parts[0].text,
    tokens: data.usageMetadata?.totalTokenCount || 0,
  };

  cache.set(cacheKey, result);
  return result;
}

async function callClaude(messages, apiKey) {
  const cacheKey = `claude-${getCacheKey(messages)}`;
  if (cache.has(cacheKey)) return cache.get(cacheKey);

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 2000,
      messages,
    }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error?.message || 'Claude error');

  const result = {
    provider: 'claude',
    content: data.content[0].text,
    tokens: data.usage.input_tokens + data.usage.output_tokens,
  };

  cache.set(cacheKey, result);
  return result;
}

app.post('/api/ensemble', async (req, res) => {
  try {
    const { messages, providers: providerIds } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Invalid messages format' });
    }

    const providers = {
      groq: process.env.GROQ_API_KEY,
      gemini: process.env.GEMINI_API_KEY,
      claude: process.env.CLAUDE_API_KEY,
    };

    const promises = providerIds
      .filter(id => providers[id])
      .map(async (id) => {
        try {
          if (id === 'groq') return await callGroq(messages, providers[id]);
          if (id === 'gemini') return await callGemini(messages, providers[id]);
          if (id === 'claude') return await callClaude(messages, providers[id]);
        } catch (error) {
          return { provider: id, error: error.message };
        }
      });

    const responses = await Promise.all(promises);
    const valid = responses.filter(r => !r.error);

    if (valid.length === 0) {
      return res.status(503).json({ 
        error: 'All LLM providers failed',
        details: responses 
      });
    }

    res.json({
      responses: valid,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    providers: {
      groq: !!process.env.GROQ_API_KEY,
      gemini: !!process.env.GEMINI_API_KEY,
      claude: !!process.env.CLAUDE_API_KEY,
    },
  });
});

app.listen(PORT, () => {
  console.log(`✅ Multi-LLM Backend running on port ${PORT}`);
  console.log(`📊 Health check: GET /health`);
  console.log(`🎯 Ensemble endpoint: POST /api/ensemble`);
});
