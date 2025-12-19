/** @type {import('tailwindcss').Config} */
module.exports = {
  // CRITICAL: Tells Tailwind to scan all React files for class names
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}