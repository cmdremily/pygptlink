import re


class SentenceExtractor:
    def extract_partial(self, partial_input: str) -> tuple[list[str], str]:
        # Chat GPT doesn't break lines to any particular line width, relying
        # on word wrapping to do the right thing. Therefore, any embedded \n
        # are natural splitting points.
        lines = partial_input.splitlines()
        trimmed_lines = [line for line in lines if line.strip()]

        if len(trimmed_lines) == 0:
            return [], ''
        elif partial_input.endswith('\n'):
            return [line.strip() for line in trimmed_lines], ''
        else:
            full_lines = trimmed_lines[:-1]
            partial_tail = trimmed_lines[-1]

            # Split partial_tail on punctuation marks
            sentences = re.split(r'(?<=[.;?!])\s+', partial_tail)

            # Remove empty strings from sentences list
            sentences = [
                sentence for sentence in sentences if sentence.lstrip()]

            # Add sentences to full_lines
            full_lines.extend(sentences[0:-1])

            # if len(full_lines):
            #    print(f"Full: {full_lines}, partial: {sentences[-1]}")

            return [line.strip() for line in full_lines], sentences[-1]
