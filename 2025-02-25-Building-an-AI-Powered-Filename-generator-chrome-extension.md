---
layout: post
title:  Building an AI-Powered Filename Generator Chrome Extension
date:   2025-02-25 07:42:44 -0500
description: "Learn how to create a Chrome extension that uses AI to automatically generate meaningful filenames when downloading files, with full code examples and implementation details."
---
# Building an AI-Powered Filename Generator Chrome Extension

[Chrome Web Store](https://chromewebstore.google.com/detail/ai-filename-generator/eocbkbnabbmclgneeakdbglicbhbimbj)

## Introduction

Managing files efficiently can be a challenge, especially when dealing with vague or cluttered filenames. To solve this, I developed the **AI Filename Generator**—a Chrome extension that intelligently renames files based on their content. This blog post will walk you through how I built it using my open-source repository: [chrome-ai-filename-generator](https://github.com/kliewerdaniel/chrome-ai-filename-generator).

You can also install the extension directly from the Chrome Web Store: [AI Filename Generator](https://chromewebstore.google.com/detail/ai-filename-generator/eocbkbnabbmclgneeakdbglicbhbimbj).

By the end of this post, you'll understand the core technologies used, how to set up the extension, and the process of integrating AI into a simple yet effective Chrome tool.

---

## Why Build an AI Filename Generator?

I often download files with generic names like `document.pdf`, `image123.jpg`, or `scan_2024.png`. Instead of manually renaming them, I wanted a tool that could:

- Analyze the file contents (text or metadata)
- Generate meaningful, structured filenames automatically
- Seamlessly integrate into Chrome’s download flow

This extension enhances productivity by making file organization smarter and faster.

---

## Tech Stack Overview

This Chrome extension is built using:

- **Manifest v3** – The latest Chrome extension framework
- **JavaScript & HTML/CSS** – For frontend interactions
- **OpenAI API** (or local AI models) – For intelligent filename generation
- **Chrome Downloads API** – To modify filenames upon download
- **Webpack & Babel** – For modern JavaScript compilation

---

## How It Works

The AI Filename Generator intercepts file downloads and renames them using AI-generated suggestions. Here’s the high-level workflow:

1. **Intercept a File Download**
   - Using Chrome’s `downloads.onDeterminingFilename` API, the extension listens for download events.
2. **Analyze File Metadata**
   - Extracts information like file type, source URL, and content (if accessible).
3. **Send Data to AI Model**
   - Requests a relevant filename based on context.
4. **Rename the File**
   - Modifies the filename before saving it to disk.

---

## Setting Up the Extension

Want to try it out or contribute? Follow these steps:

### 1. Clone the Repository

```sh
git clone https://github.com/kliewerdaniel/chrome-ai-filename-generator.git
cd chrome-ai-filename-generator
```

### 2. Install Dependencies

```sh
npm install
```

### 3. Build the Extension

```sh
npm run build
```

### 4. Load the Extension in Chrome

1. Open `chrome://extensions/`
2. Enable **Developer Mode** (top-right corner)
3. Click **Load Unpacked** and select the `dist/` folder

---

## Key Features Explained

### 1. **Intercepting Downloads**

```javascript
chrome.downloads.onDeterminingFilename.addListener((downloadItem, suggest) => {
  const originalFilename = downloadItem.filename;
  getAIEnhancedFilename(originalFilename).then((newFilename) => {
    suggest({ filename: newFilename });
  });
});
```

This snippet listens for file downloads and passes the filename to our AI function.

### 2. **Generating AI-Based Filenames**

```javascript
async function getAIEnhancedFilename(originalName) {
  const response = await fetch("https://api.openai.com/v1/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4",
      prompt: `Suggest a meaningful filename for: ${originalName}`,
      max_tokens: 10
    })
  });
  const data = await response.json();
  return data.choices[0].text.trim();
}
```

This function calls OpenAI’s API to generate a more descriptive filename based on the original one.

---

## Future Improvements

While the current version is functional, there are some enhancements I plan to explore:

- **Local LLM Support** – Allowing users to run filename suggestions without an internet connection.
- **Content-Based Naming** – Extracting text from PDFs/images to generate even more accurate filenames.
- **Customization Options** – Letting users define filename formats (e.g., date-based, project-based).

---

## Conclusion

The AI Filename Generator Chrome extension is a small but powerful tool that enhances file organization. By leveraging AI, we can automate mundane tasks like renaming files, ultimately improving productivity. If you're interested, check out the [GitHub repo](https://github.com/kliewerdaniel/chrome-ai-filename-generator) and feel free to contribute!

You can also install the extension directly from the Chrome Web Store: [AI Filename Generator](https://chromewebstore.google.com/detail/ai-filename-generator/eocbkbnabbmclgneeakdbglicbhbimbj).

What features would you like to see added? Let me know in the comments!
