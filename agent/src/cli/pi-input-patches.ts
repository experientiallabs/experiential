/**
 * Input checks in the pinned Pi packages: a blank -p prompt and a prompt larger than the model's
 * input budget are refused before any call. pi-patches.ts applies these with its other edits (same
 * exact-match rules, same tests).
 */

/** Refuse a print or JSON run whose prompt is missing or only whitespace (main.js). */
export const EMPTY_PROMPT_GUARD = `    if (appMode === "print" || appMode === "json") {
        const blank = (text) => text === undefined || text.trim() === "";
        const noPrompt = blank(initialMessage) && initialImages === undefined;
        if (noPrompt || parsed.messages.some(blank)) {
            const what = noPrompt && initialMessage === undefined && parsed.messages.length === 0 ? "no prompt was given" : "the prompt is empty (only whitespace)";
            console.error(chalk.red(\`Error: \${what}, so there is nothing to send.\`));
            console.error('Usage: exp-code -p "<prompt>"   (or pipe it on stdin: echo "<prompt>" | exp-code -p)');
            console.error("Run exp-code without -p for the interactive UI, or exp-code --help for every option.");
            process.exit(2);
        }
    }
`;

/**
 * A prompt too large for the model's input budget (core/agent-session.js). A 1 MB paste or piped
 * file was sent whole and refused by the provider (or cut by the gateway) after it was billed. The
 * prompt alone is estimated as Pi estimates context (four characters a token) and refused before
 * any call when it exceeds the window less the model's output reservation, which the provider
 * counts inside the window.
 */
export const PROMPT_SIZE_GUARD = `/** exp-code: why a prompt cannot fit the model's input budget, or undefined when it may. */
function expPromptTooLarge(text, model) {
    if (!model || !(model.contextWindow > 0))
        return undefined;
    const budget = model.contextWindow - (model.maxTokens > 0 ? model.maxTokens : 0);
    const tokens = Math.ceil(text.length / 4);
    if (budget <= 0 || tokens <= budget)
        return undefined;
    const count = (n) => n.toLocaleString("en-US");
    return \`This prompt is too large for \${model.id}: about \${count(tokens)} tokens (\${count(text.length)} characters), and the model takes at most \${count(budget)} tokens of input (its \${count(model.contextWindow)}-token window less \${count(model.contextWindow - budget)} reserved for the answer). Nothing was sent. Shorten it, name the file for the agent to read in parts instead of pasting or piping it whole, or pick a model with a larger window (/model, or exp-code --list-models).\`;
}
`;
