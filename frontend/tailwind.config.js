/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0f17",
        panel: "#111827",
        panel2: "#0f1623",
        line: "#1f2937",
        muted: "#8b95a7",
        ink: "#e5e7eb",
        accent: "#38bdf8",
        lvl: { gray: "#6b7280", green: "#22c55e", yellow: "#eab308", orange: "#f97316", red: "#ef4444" },
      },
      fontFamily: { sans: ["Inter", "Noto Kufi Arabic", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
    },
  },
  plugins: [],
};
