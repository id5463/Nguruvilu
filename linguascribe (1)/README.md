# LinguaScribe - Real-time Translation & Contextual Note-Taking

LinguaScribe is an intelligent web application that provides real-time language translation and automatically organizes the translated content into context-aware notes. It supports speech input and UI localization.

## Features

*   Real-time translation of typed or spoken language.
*   Contextual note generation from translated text, triggered manually.
*   Support for multiple source, target, and UI languages.
*   Speech recognition for hands-free input.
*   "Real-time" and "Composed" translation modes for different input styles.
*   Session History: Save, load, and manage past work sessions with automatically generated titles.
*   Persistent storage of text, notes, and settings using browser localStorage.
*   Responsive design for various screen sizes.

## Prerequisites (For Local Development)

Before you begin, ensure you have the following installed on your system. These instructions assume you are starting with a relatively fresh development environment.

1.  **Node.js (which includes npm)**:
    *   **Why?**: Node.js is essential for running modern JavaScript projects, managing dependencies, and using development servers. npm (Node Package Manager) comes with Node.js and is used to install tools.
    *   **How?**: Download and install from [nodejs.org](https://nodejs.org/).

2.  **A Code Editor**:
    *   **Why?**: To view and edit the project files.
    *   **Examples**: Visual Studio Code (VS Code), Sublime Text, WebStorm.

3.  **Gemini API Key**:
    *   **Why?**: The application relies on the Google Gemini API for translation and AI-powered summarization.
    *   **How?**: Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

4.  **A Development Server (e.g., Vite)**:
    *   **Why?**:
        *   The project uses `.tsx` files (TypeScript with React JSX), which browsers cannot run directly. A development server transpiles (converts) these into standard JavaScript.
        *   It handles environment variables (like your API Key) and makes them accessible to the client-side code.
        *   It provides a local web server to serve your `index.html` and other assets.
    *   **How (Example with Vite)**: Vite is a fast and modern build tool. You can install it in your project using npm (which you get with Node.js). You would typically initialize a Vite project or add Vite to an existing one.
        *   To add Vite to an existing project directory (where your files are):
            ```bash
            # Navigate to your project directory in the terminal
            cd path/to/your/linguascribe-project
            # Initialize a package.json if you don't have one
            npm init -y
            # Install Vite and the React plugin
            npm install vite @vitejs/plugin-react --save-dev
            ```
        *   You would then need a `vite.config.js` (or `.ts`) file and ensure your `index.html` is the entry point. Vite's official documentation provides excellent guidance.
        *   A simpler approach if starting "fresh" but having the LinguaScribe files is to create a new Vite project and then copy LinguaScribe's `src`, `public` (if any, for assets like `locales`), and `index.html` (modifying its script tags if Vite uses a different entry point like `src/main.tsx` by default) into the new Vite project structure.

5.  **`cloudflared` command-line tool (Optional)**:
    *   **Why?**: Only needed if you plan to expose your local server to the internet using Cloudflare Tunnel (as detailed further down).
    *   **How?**: Download and install from the official Cloudflare documentation: [Install cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)

## Local Development & Serving

These instructions guide you through configuring and running the application on your local machine.

### 1. Project Files

Ensure all the provided project files (`index.html`, `App.tsx`, `components/`, `services/`, `locales/`, `contexts/`, `utils/`, `types.ts`, `constants.ts`) are organized correctly in your chosen project directory.

### 2. Configuration - API Key

*   The application expects the Gemini API key to be available as `process.env.API_KEY` in `services/geminiService.ts`.
*   **Create a `.env` file**: In the root of your project directory, create a file named `.env`.
*   **Add your API Key**: Open the `.env` file and add your Gemini API Key in the following format:
    ```env
    API_KEY=YOUR_GEMINI_API_KEY_HERE
    ```
    Replace `YOUR_GEMINI_API_KEY_HERE` with your actual API key obtained from Google AI Studio.
*   **Note**: Your development server (e.g., Vite) is responsible for loading this `.env` file and making the `API_KEY` available to your application code.

### 3. Running the Application Locally

1.  **Install Project Dependencies (if using a package manager like npm with Vite)**:
    *   If you've set up your project with Vite and have a `package.json` (e.g., by running `npm init -y` and `npm install vite @vitejs/plugin-react --save-dev`), you might have other dependencies to install, although the core libraries (React, @google/genai) are loaded via CDN in the current `index.html`. If you switch to managing these via npm, you'd run:
        ```bash
        npm install
        ```

2.  **Start your development server**:
    *   The command depends on how your development server is configured. If you set up Vite and added a `dev` script to your `package.json` (e.g., `"dev": "vite"`), you would run:
        ```bash
        npm run dev
        ```
    *   If using Vite directly without a script:
        ```bash
        npx vite
        ```
    *   For other servers like Parcel:
        ```bash
        npx parcel index.html
        ```

3.  **Access the Application**:
    *   Once the server starts, it will typically output a local URL to the terminal, such as `http://localhost:5173` (common for Vite) or `http://localhost:3000`.
    *   Open this URL in your web browser.
    *   Make a note of this exact URL and port number (e.g., `http://localhost:PORT_NUMBER`). You'll need it if you proceed with the Cloudflare Tunnel setup.

## Exposing Locally with Cloudflare Tunnel (Optional)

Cloudflare Tunnel allows you to securely expose your local web server to the internet without needing a public IP address or complex firewall configurations.

### Prerequisites for Tunneling

*   A **Cloudflare Account** (free to sign up).
*   A **Domain Name** managed through Cloudflare (optional, but recommended for a custom, memorable URL). `cloudflared` will provide a random subdomain if you don't use your own.
*   Your local application server must be running (see previous section).
*   The `cloudflared` command-line tool must be installed (see "Prerequisites (For Local Development)").

### Steps

1.  **Login to Cloudflare via `cloudflared`**:
    *   Open your terminal or command prompt.
    *   Run:
        ```bash
        cloudflared login
        ```
    *   This will open a browser window asking you to log in to your Cloudflare account and authorize `cloudflared`. Select a website (domain) you want to associate the tunnel with if prompted.

2.  **Create a Tunnel**:
    *   Run the following command. You can replace `linguascribe-tunnel` with any name you prefer:
        ```bash
        cloudflared tunnel create linguascribe-tunnel
        ```
    *   This command outputs information, including your **Tunnel ID** (a UUID like `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`) and creates a credential file. **Copy this Tunnel ID.**

3.  **Configure DNS for your Tunnel (Optional, for a custom domain like `linguascribe.yourdomain.com`)**:
    *   If you want to use a custom subdomain:
        *   Go to your Cloudflare dashboard, select your domain.
        *   Navigate to the "DNS" records page.
        *   Create a **CNAME** record:
            *   **Type**: `CNAME`
            *   **Name**: Your desired subdomain (e.g., `linguascribe` if your domain is `yourdomain.com`).
            *   **Target**: Your `YOUR_TUNNEL_ID.cfargotunnel.com`. (Replace `YOUR_TUNNEL_ID` with the ID from the previous step).
            *   **Proxy status**: Ensure it's proxied (orange cloud). Click "Save".

4.  **Run the Tunnel**:
    *   Ensure your local LinguaScribe development server is running (e.g., at `http://localhost:5173`).
    *   In your terminal, run the following command.
        *   Replace `YOUR_LOCAL_APP_URL` with the actual URL of your local server (e.g., `http://localhost:5173`).
        *   Replace `YOUR_TUNNEL_NAME_OR_ID` with the name you chose (e.g., `linguascribe-tunnel`) or the Tunnel ID.
        ```bash
        cloudflared tunnel run --url YOUR_LOCAL_APP_URL YOUR_TUNNEL_NAME_OR_ID
        ```
        *Example if your local app is on port 5173 and tunnel name is `linguascribe-tunnel`*:
        ```bash
        cloudflared tunnel run --url http://localhost:5173 linguascribe-tunnel
        ```

5.  **Access Your Application Publicly**:
    *   `cloudflared` will show logs indicating the tunnel is active.
    *   Your LinguaScribe application should now be accessible via:
        *   The `*.cfargotunnel.com` URL assigned when you created the tunnel (if you didn't use a custom domain).
        *   Your custom domain (e.g., `https://linguascribe.yourdomain.com`) if you configured DNS.

### Placeholders to Configure for Cloudflare Tunnel:

*   `http://localhost:PORT_NUMBER` (or `YOUR_LOCAL_APP_URL`): The full URL of your locally running application.
*   `linguascribe-tunnel` (or `YOUR_TUNNEL_NAME_OR_ID`): The name or ID you assign to your Cloudflare Tunnel.
*   `yourdomain.com`: If using a custom domain, replace this with your actual domain registered with Cloudflare.
*   `YOUR_TUNNEL_ID`: The unique ID generated when you create a tunnel, used for the CNAME target.

## Important Security Note for Public Deployment

The current setup, where the Gemini API key might be included in client-side JavaScript (even if injected by a dev server via `process.env`), is **not secure for production or public deployment**. If your JavaScript files are accessible, your API key could be exposed.

For a more secure public deployment:

*   **Backend Proxy**: It's highly recommended to proxy your Gemini API calls through a backend service. This backend service would securely store the API key and make requests to the Gemini API on behalf of the client. Your client-side application would then call your backend proxy.
    *   Examples: A simple Node.js/Express server, Python/Flask server, or serverless functions (like Cloudflare Workers, AWS Lambda, Google Cloud Functions).
*   **Cloudflare Workers**: Since you're using Cloudflare Tunnel, Cloudflare Workers are an excellent option for creating this backend proxy. You can store the API key as a secret in the Worker.

By using a backend proxy, your API key is never exposed to the user's browser, significantly improving security.
