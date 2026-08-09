import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
/* Bundled, not fetched: this app runs against a local backend and is expected
   to work with no internet at all, so a Google Fonts <link> would silently fall
   back to Segoe UI on exactly the machines that matter. The variable file is one
   request and covers every weight the UI uses. */
import '@fontsource-variable/inter'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
