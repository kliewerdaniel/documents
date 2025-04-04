---
layout: post
title:  Quiz Platform
date:   2025-03-09 11:42:44 -0500
---
# Building a High-Performance Quiz Platform with Next.js and Firebase

Creating an online quiz platform can be challenging, especially when you need to handle user authentication, store scores, and display dynamic content. In this comprehensive guide, I'll walk you through how to optimize a Next.js quiz application that leverages Firebase for authentication and Firestore for data storage.

## The Challenge of Quiz Applications

Many educational platforms struggle with performance issues, security vulnerabilities, and code maintainability when implementing quiz functionality. Whether you're building a learning management system, an educational app, or just a fun quiz site, these challenges can significantly impact user experience.

Let's explore how to streamline a Next.js quiz platform with Firebase integration to create a secure, fast, and maintainable solution.

## 1. Centralizing Firebase Initialization

One common mistake is initializing Firebase multiple times across different components. This not only affects performance but can also lead to unexpected behaviors.

### The Solution: Single Firebase Instance

Create a dedicated `firebase.js` file to handle initialization once:

```javascript
// firebase.js
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: 'YOUR_KEY',
  authDomain: 'your-app.firebaseapp.com',
  projectId: 'your-app',
  storageBucket: 'your-app.appspot.com',
  messagingSenderId: '123456789',
  appId: '1:123456789:web:abcdef123456789'
};

// Initialize Firebase only once
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

export { auth, db };
```

By exporting the initialized `auth` and `db` instances, you can import them wherever needed without creating redundant Firebase connections.

## 2. Optimizing the Quiz Page Component

Your quiz page should efficiently handle quiz rendering and score saving without unnecessary re-renders or network calls.

### Implementation Approach #1: Direct HTML Rendering

```javascript
// pages/quiz/[id].js
import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { doc, setDoc } from 'firebase/firestore';
import { auth, db } from '../../firebase';

const QuizPage = ({ quizHtml }) => {
  const router = useRouter();
  const { id } = router.query;

  useEffect(() => {
    // Expose the saveScore function to the quiz content
    const saveScore = async (score) => {
      const user = auth.currentUser;
      if (user) {
        const userDoc = doc(db, 'grades', user.uid);
        await setDoc(
          userDoc,
          { [id]: { score, updatedAt: new Date() } },
          { merge: true }
        );
      }
    };

    window.saveScore = saveScore;
  }, [id]);

  return (
    <div>
      <div dangerouslySetInnerHTML={{ __html: quizHtml }} />
    </div>
  );
};

export async function getStaticProps({ params }) {
  const quizHtml = await getQuizHTML(params.id); // Implement this function to fetch quiz HTML
  return { props: { quizHtml } };
}

export async function getStaticPaths() {
  // Implement this function to generate paths for all quizzes
  return {
    paths: [
      { params: { id: 'quiz1' } },
      { params: { id: 'quiz2' } },
      // Add more quizzes as needed
    ],
    fallback: false
  };
}

export default QuizPage;
```

This approach works but has potential security risks due to the use of `dangerouslySetInnerHTML`.

## 3. Enhancing Security with Iframe Isolation

A more secure approach is to isolate quiz content within an iframe, preventing potential XSS attacks and providing better content separation.

### Implementation Approach #2: Iframe Isolation

```javascript
// pages/quiz/[id].js
import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { doc, setDoc } from 'firebase/firestore';
import { auth, db } from '../../firebase';

const QuizPage = ({ quizPath }) => {
  const router = useRouter();
  const { id } = router.query;

  useEffect(() => {
    const saveScore = async (score) => {
      const user = auth.currentUser;
      if (user) {
        const userDoc = doc(db, 'grades', user.uid);
        await setDoc(
          userDoc,
          { [id]: { score, updatedAt: new Date() } },
          { merge: true }
        );
      } else {
        // Handle unauthenticated user scenario
        console.log('User not authenticated. Score not saved.');
        router.push('/login?returnUrl=' + router.asPath);
      }
    };

    // Listen for messages from the iframe
    window.addEventListener('message', (event) => {
      if (event.data.type === 'saveScore') {
        saveScore(event.data.score);
      }
    });

    // Cleanup event listener
    return () => {
      window.removeEventListener('message', (event) => {
        if (event.data.type === 'saveScore') {
          saveScore(event.data.score);
        }
      });
    };
  }, [id, router]);

  return (
    <div className="quiz-container">
      <h1>Quiz {id}</h1>
      <iframe
        src={quizPath}
        width="100%"
        height="600px"
        style={{ border: 'none' }}
        title={`Quiz ${id}`}
      />
    </div>
  );
};

export async function getStaticProps({ params }) {
  const quizPath = `/quizzes/${params.id}.html`; // Path to quiz HTML files in public directory
  return { props: { quizPath } };
}

export async function getStaticPaths() {
  // Generate paths for all quizzes
  return {
    paths: [
      { params: { id: 'quiz1' } },
      { params: { id: 'quiz2' } },
      // Add more quizzes as needed
    ],
    fallback: false
  };
}

export default QuizPage;
```

To make this approach work, your quiz HTML files (stored in the `public/quizzes/` directory) should include code to communicate with the parent page:

```html
<!-- public/quizzes/quiz1.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Quiz 1</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    .question { margin-bottom: 20px; }
    button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; }
  </style>
</head>
<body>
  <h2>Science Quiz</h2>
  <form id="quizForm" onsubmit="calculateScore(); return false;">
    <div class="question">
      <p>1. What is the chemical symbol for water?</p>
      <input type="radio" name="q1" value="a" id="q1a">
      <label for="q1a">O2</label><br>
      <input type="radio" name="q1" value="b" id="q1b">
      <label for="q1b">H2O</label><br>
      <input type="radio" name="q1" value="c" id="q1c">
      <label for="q1c">CO2</label>
    </div>
    
    <div class="question">
      <p>2. Which planet is known as the Red Planet?</p>
      <input type="radio" name="q2" value="a" id="q2a">
      <label for="q2a">Venus</label><br>
      <input type="radio" name="q2" value="b" id="q2b">
      <label for="q2b">Mars</label><br>
      <input type="radio" name="q2" value="c" id="q2c">
      <label for="q2c">Jupiter</label>
    </div>
    
    <button type="submit">Submit Quiz</button>
  </form>

  <script>
    function calculateScore() {
      const form = document.getElementById('quizForm');
      let score = 0;
      const answers = {
        q1: 'b', // H2O
        q2: 'b'  // Mars
      };
      
      // Check each question
      for (const [question, correctAnswer] of Object.entries(answers)) {
        const selectedValue = form.elements[question].value;
        if (selectedValue === correctAnswer) {
          score += 1;
        }
      }
      
      const totalQuestions = Object.keys(answers).length;
      const percentage = Math.round((score / totalQuestions) * 100);
      
      // Send score to parent page
      window.parent.postMessage({ type: 'saveScore', score: percentage }, '*');
      
      // Show results to user
      alert(`You scored ${percentage}% (${score}/${totalQuestions} correct)`);
    }
  </script>
</body>
</html>
```

## 4. Security Considerations

Even with the iframe approach, there are additional security measures to consider:

### Content Security Policy (CSP)

Set up a proper Content Security Policy in your Next.js application:

```javascript
// next.config.js
module.exports = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self'; frame-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
          }
        ]
      }
    ]
  }
}
```

### HTML Sanitization

If you're using the direct HTML rendering approach, always sanitize any external HTML content:

```javascript
import DOMPurify from 'dompurify';

// In your component
const sanitizedHtml = DOMPurify.sanitize(quizHtml);

return (
  <div dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />
);
```

## 5. Enhancing Your Quiz Platform

To create a truly robust quiz platform, consider these additional features:

### Authentication UI

Integrate Firebase Authentication UI for a seamless login experience:

```javascript
// components/AuthUI.js
import { useState, useEffect } from 'react';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import { auth } from '../firebase';

const AuthUI = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const handleSignOut = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div className="auth-ui">
      {user ? (
        <div>
          <p>Welcome, {user.displayName || user.email}</p>
          <button onClick={handleSignOut}>Sign Out</button>
        </div>
      ) : (
        <button onClick={() => window.location.href = '/login'}>Sign In</button>
      )}
    </div>
  );
};

export default AuthUI;
```

### Score Dashboard

Create a dashboard to display users' quiz scores:

```javascript
// pages/dashboard.js
import { useState, useEffect } from 'react';
import { doc, getDoc } from 'firebase/firestore';
import { onAuthStateChanged } from 'firebase/auth';
import { auth, db } from '../firebase';

const Dashboard = () => {
  const [scores, setScores] = useState({});
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      setUser(currentUser);
      
      if (currentUser) {
        try {
          const userDocRef = doc(db, 'grades', currentUser.uid);
          const userDoc = await getDoc(userDocRef);
          
          if (userDoc.exists()) {
            setScores(userDoc.data());
          }
        } catch (error) {
          console.error('Error fetching scores:', error);
        } finally {
          setLoading(false);
        }
      } else {
        setLoading(false);
      }
    });
    
    return unsubscribe;
  }, []);

  if (loading) return <div>Loading your scores...</div>;
  
  if (!user) return <div>Please log in to view your dashboard</div>;

  return (
    <div className="dashboard">
      <h1>Your Quiz Scores</h1>
      
      {Object.keys(scores).length > 0 ? (
        <ul className="scores-list">
          {Object.entries(scores).map(([quizId, data]) => (
            <li key={quizId} className="score-item">
              <div className="quiz-name">Quiz: {quizId}</div>
              <div className="quiz-score">Score: {data.score}%</div>
              <div className="quiz-date">
                Completed: {new Date(data.updatedAt.toDate()).toLocaleString()}
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p>You haven't completed any quizzes yet.</p>
      )}
    </div>
  );
};

export default Dashboard;
```

### Admin Quiz Management

For educators or administrators, implement a quiz management system:

```javascript
// pages/admin/quizzes.js
import { useState } from 'react';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { storage } from '../../firebase';

const QuizManagement = () => {
  const [file, setFile] = useState(null);
  const [quizId, setQuizId] = useState('');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!file || !quizId) {
      setMessage('Please select a file and provide a quiz ID');
      return;
    }
    
    setUploading(true);
    setMessage('');
    
    try {
      // Upload to Firebase Storage
      const storageRef = ref(storage, `quizzes/${quizId}.html`);
      await uploadBytes(storageRef, file);
      const downloadURL = await getDownloadURL(storageRef);
      
      setMessage(`Quiz uploaded successfully! URL: ${downloadURL}`);
      setFile(null);
      setQuizId('');
    } catch (error) {
      console.error('Error uploading quiz:', error);
      setMessage(`Error uploading quiz: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="admin-panel">
      <h1>Quiz Management</h1>
      
      <form onSubmit={handleUpload} className="upload-form">
        <div className="form-group">
          <label htmlFor="quizId">Quiz ID:</label>
          <input
            type="text"
            id="quizId"
            value={quizId}
            onChange={(e) => setQuizId(e.target.value)}
            placeholder="e.g., science-quiz-1"
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="quizFile">Quiz HTML File:</label>
          <input
            type="file"
            id="quizFile"
            accept=".html"
            onChange={handleFileChange}
            required
          />
        </div>
        
        <button type="submit" disabled={uploading}>
          {uploading ? 'Uploading...' : 'Upload Quiz'}
        </button>
      </form>
      
      {message && <div className="message">{message}</div>}
    </div>
  );
};

export default QuizManagement;
```

## Performance Optimization

To ensure your quiz platform runs smoothly, implement these additional optimizations:

1. **Implement caching**: Use Firestore's offline capabilities to allow users to take quizzes without an internet connection.

2. **Lazy loading**: Only load quiz content when necessary to reduce initial page load times.

3. **Server-side rendering for dashboard pages**: Pre-render data-heavy pages to improve perceived performance.

4. **Implement analytics**: Track quiz completion rates and user engagement to identify areas for improvement.

## Conclusion

Building a high-performance quiz platform with Next.js and Firebase requires careful planning and implementation. By centralizing Firebase initialization, securing content with iframes or proper sanitization, and implementing additional features like user dashboards and admin panels, you can create a robust, maintainable quiz application.

The approaches outlined in this guide will help you avoid common pitfalls, enhance security, and provide a seamless experience for both students and educators. Whether you're building an educational platform, a corporate training tool, or just a fun quiz site, these techniques will help you create a professional-grade solution.

Remember that security should always be a priority when handling user data and displaying dynamic content. Regularly audit your code and stay updated with the latest security best practices for web applications.
