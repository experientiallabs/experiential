import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        surface: {
          DEFAULT: "var(--surface)",
          subtle: "var(--surface-subtle)"
        },
        ink: "var(--ink)",
        muted: {
          DEFAULT: "var(--muted)",
          2: "var(--muted-2)"
        },
        line: {
          DEFAULT: "var(--line)",
          strong: "var(--line-strong)"
        },
        success: {
          DEFAULT: "var(--success)",
          soft: "var(--success-soft)"
        },
        warning: {
          DEFAULT: "var(--warning)",
          soft: "var(--warning-soft)"
        },
        danger: {
          DEFAULT: "var(--danger)",
          soft: "var(--danger-soft)"
        },
        purple: {
          DEFAULT: "var(--purple)",
          soft: "var(--purple-soft)"
        },
        accent: "var(--accent)",
        hover: "var(--hover)",
        active: "var(--active)"
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "-apple-system", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"]
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)"
      }
    }
  },
  plugins: []
};

export default config;
