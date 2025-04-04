---
layout: post
title:  Learning from the Past
date:   2024-12-14 07:42:44 -0500
---

### Learning From the Past and Building a Better Future Through Technology

Our world stands at a critical juncture. We’ve witnessed how the unchecked pursuit of strategic advantage, from mid-20th century conflicts to the present day, can normalize moral compromises. The history of warfare, internment camps, and the nuclear arms race taught us a lesson: the ends do not justify the means. Yet here we are again, seeing advanced technologies—drones, AI, cryptographic frameworks—funneled into machines of war. We see hypocrisy in how international rules are applied selectively, and we see how the very frameworks meant to maintain peace can be bent or broken for short-term gain.

But technology doesn’t have to serve destruction. Just as the same drone technology can be repurposed to improve healthcare delivery in remote areas, or augmented reality can train doctors more efficiently, the tools we create can heal rather than harm. This choice—how we apply our technology—is ours to make. We can build a future where AI supports better communication, encourages empathy, and guides us toward more conscientious behavior.

This brings us to today’s project: a small web application that uses machine learning and a graph-based data structure to help people communicate more positively. Instead of guiding deadly precision strikes, this codebase is designed to guide more constructive dialogue. By highlighting and mapping out potentially hurtful language, the app nudges us toward healthier, more uplifting forms of expression.

We’re acknowledging our past failures and choosing a different path forward. This app, while small and symbolic, is a testament to the idea that we can use the most advanced tools at our disposal to cultivate empathy rather than enmity. We can support each other by learning from the past and building a kinder digital world, one line of code at a time.

---

### Guide: Building the “PositiveWords Graph” Application

**Goal:**  
Set up a React application that integrates a toxicity-detection ML model and visualizes user input as a weighted graph of words. This encourages more thoughtful communication and leverages technology to improve the world in a small but meaningful way.

**Key Features:**  
- A React front-end that allows the user to enter a message.  
- Integration with a pre-trained TensorFlow.js toxicity model to detect harmful language.  
- A graph representation of the user’s text where nodes are words and edges represent adjacency and frequency, highlighting potentially problematic areas.  
- Deployment capability via Git and Netlify so changes can be easily pushed live.

#### Prerequisites
- **Node.js and npm** installed (verify with `node -v` and `npm -v`)
- **Git** installed (verify with `git --version`)
- A **GitHub** account for version control
- A **Netlify** account for free deployment

#### Step-by-Step Instructions

**1. Create a New React App**  
Use `create-react-app` for quick setup.

```bash
# Navigate to your projects directory
cd /path/to/projects

# Create a new React app
npx create-react-app positivewords-graph
```

**2. Move Into the Project and Install Dependencies**  
```bash
cd positivewords-graph
npm install @tensorflow/tfjs @tensorflow-models/toxicity
```

**3. Replace the Default Code With Our Custom Code**  
- Open the project in your code editor.
- Replace `src/App.js` and `src/App.css` with the provided code below.
- Ensure `src/index.js` and `package.json` match the provided snippets.

**`package.json` (already created by create-react-app, just ensure dependencies are present):**
```json
{
  "name": "positivewords-graph",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@tensorflow-models/toxicity": "^1.2.2",
    "@tensorflow/tfjs": "^4.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "react-scripts": "5.0.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  }
}
```

**`src/index.js`:**
```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './App.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```

**`src/App.js`:**
```javascript
import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { load } from '@tensorflow-models/toxicity';
import './App.css';

function buildGraphFromText(text, toxicWords) {
  const words = text
    .toLowerCase()
    .replace(/[^\w\s]/gi, '')
    .split(/\s+/)
    .filter(w => w.trim().length > 0);
  
  const nodes = {};
  const edges = {};

  words.forEach(w => {
    if (!nodes[w]) {
      nodes[w] = { word: w, toxicityWeight: toxicWords.includes(w) ? 1 : 0 };
    }
  });

  for (let i = 0; i < words.length - 1; i++) {
    const a = words[i];
    const b = words[i + 1];
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    if (!edges[key]) {
      edges[key] = { a, b, weight: 0 };
    }
    edges[key].weight += 1;
  }

  return { nodes: Object.values(nodes), edges: Object.values(edges) };
}

function App() {
  const [model, setModel] = useState(null);
  const [inputText, setInputText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const threshold = 0.9;
  
  useEffect(() => {
    load(threshold).then(m => {
      setModel(m);
    });
  }, [threshold]);

  const analyzeText = async () => {
    if (!model || !inputText) return;
    const predictions = await model.classify([inputText]);
    setAnalysis(predictions);
  };

  const handleChange = (e) => {
    setInputText(e.target.value);
  };

  const getToxicWords = () => {
    if (!analysis) return [];
    const toxicLabels = analysis.filter(pred => pred.results[0].match === true);
    if (toxicLabels.length === 0) return [];
    const words = inputText
      .toLowerCase()
      .replace(/[^\w\s]/gi, '')
      .split(/\s+/)
      .filter(w => w.trim().length > 0);
    return toxicLabels.length > 0 ? words : [];
  };

  const toxicWords = getToxicWords();
  const graphData = buildGraphFromText(inputText, toxicWords);

  return (
    <div className="App">
      <header className="App-header">
        <h1>PositiveWords Graph</h1>
        <p>Encouraging healthier communication with ML and graph insights.</p>
      </header>
      <main>
        <h2>Analyze Your Message</h2>
        <textarea 
          placeholder="Type your message here..."
          value={inputText}
          onChange={handleChange}
        />
        <br />
        <button onClick={analyzeText} disabled={!model || !inputText}>
          Analyze
        </button>
        {analysis && (
          <div className="analysis-results">
            {analysis.some(a => a.results[0].match) ? (
              <div className="result negative">
                <h3>Consider Rewriting</h3>
                <p>Your message may contain harmful language. The graph below shows words detected and their relationships.</p>
              </div>
            ) : (
              <div className="result positive">
                <h3>Looks Good!</h3>
                <p>No harmful language detected. The graph below shows the words and their neutral relationships.</p>
              </div>
            )}
          </div>
        )}
        {graphData.nodes.length > 0 && (
          <div className="graph-display">
            <h3>Graph Overview</h3>
            <p><strong>Nodes:</strong> Each unique word, weighted if considered toxic.</p>
            <p><strong>Edges:</strong> Co-occurrence frequency between words.</p>
            <div className="graph-section">
              <h4>Nodes</h4>
              <ul>
                {graphData.nodes.map((node, i) => (
                  <li key={i} style={{color: node.toxicityWeight > 0 ? 'red' : 'black'}}>
                    {node.word} (toxicityWeight: {node.toxicityWeight})
                  </li>
                ))}
              </ul>
              <h4>Edges</h4>
              <ul>
                {graphData.edges.map((edge, i) => (
                  <li key={i}>
                    {edge.a} - {edge.b} (weight: {edge.weight})
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </main>
      <footer>
        <p>© 2024 PositiveWords Graph. Building a kinder world through awareness and data.</p>
      </footer>
    </div>
  );
}

export default App;
```

**`src/App.css`:**
```css
.App {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
  background: #f9f9f9;
  color: #333;
}

.App-header {
  background: #4CAF50;
  color: white;
  padding: 20px;
  text-align: center;
}

main {
  padding: 20px;
  max-width: 600px;
  margin: 0 auto;
}

main h2 {
  margin-bottom: 10px;
}

textarea {
  width: 100%;
  height: 120px;
  font-size: 16px;
  padding: 10px;
  margin-bottom: 10px;
}

button {
  padding: 10px 20px;
  background: #4CAF50;
  border: none;
  color: white;
  font-size: 16px;
  cursor: pointer;
}

button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.result {
  margin-top: 20px;
  padding: 20px;
  border-radius: 5px;
}

.result.negative {
  background: #ffe0e0;
  border: 1px solid #ffcccc;
}

.result.positive {
  background: #e0ffe0;
  border: 1px solid #ccffcc;
}

.analysis-results {
  margin-top: 20px;
}

.graph-display {
  margin-top: 20px;
  background: #fff;
  border: 1px solid #ddd;
  padding: 15px;
  border-radius: 5px;
}

.graph-display ul {
  list-style: none;
  padding-left: 0;
}

.graph-display li {
  margin-bottom: 5px;
}

footer {
  text-align: center;
  padding: 10px;
  background: #eee;
  margin-top: 20px;
}
```

**4. Run the App Locally**  
```bash
npm start
```
Open `http://localhost:3000` in your browser. Enter text, analyze it, and see the results and graph.

**5. Initialize Git and Commit Changes**  
```bash
git init
git add .
git commit -m "Initial commit of PositiveWords Graph"
```

**6. Create a GitHub Repository and Push Code**  
- Go to GitHub, create a new repository `positivewords-graph`.
- Back in terminal:
```bash
git remote add origin https://github.com/<your-username>/positivewords-graph.git
git branch -M main
git push -u origin main
```

**7. Deploy to Netlify**  
- Log into Netlify and choose “Import from Git” → “GitHub”
- Select `positivewords-graph`
- Netlify will detect it’s a React app and set build command: `npm run build` and publish directory: `build`.
- Deploy the site.
- After building, it will be accessible at a Netlify subdomain.

**8. Continuous Updates**  
Make code changes locally, then:
```bash
git add .
git commit -m "Update"
git push origin main
```
Netlify automatically rebuilds and redeploys the updated app.

---

### Conclusion

This application merges cutting-edge AI with a graph-based data structure to encourage more thoughtful communication, standing in contrast to the technologies that have too often been harnessed for harm. By following these steps, you’ve created a tool that not only highlights negative language but also reveals patterns in how words connect, helping you reflect on your message. In doing so, you’re contributing—albeit in a small way—to shaping a future where technology guides us toward empathy and understanding, rather than conflict.