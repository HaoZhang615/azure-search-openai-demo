import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "Who has the better combined ratio in 2023, Baloise or Helvetia ?",
    "How does Baloise and Helvetia define and recognize investment property?",
    "What are the differences and similarities for the interest sensitivity between Baloise and Helvetia",
    "Which key figures or earnings items are conspicuous?"
];

const GPT4V_EXAMPLES: string[] = [
    "Who has the better combined ratio in 2023, Baloise or Helvetia ?",
    "How does Baloise and Helvetia define and recognize investment property?",
    "What are the differences and similarities for the interest sensitivity between Baloise and Helvetia",
    "Which key figures or earnings items are conspicuous?"
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked, useGPT4V }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {(useGPT4V ? GPT4V_EXAMPLES : DEFAULT_EXAMPLES).map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
