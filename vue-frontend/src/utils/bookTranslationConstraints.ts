import {
  DEFAULT_AUTO_GLOSSARY_PROMPT,
  DEFAULT_HQ_GLOSSARY_PROMPT,
} from '@/constants'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'

export function createEmptyBookTranslationConstraints(): BookTranslationConstraints {
  return {
    glossary: {
      enabled: false,
      autoExtractEnabled: false,
      autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
      autoExtractBatchPages: 5,
      hqAutoFillEnabled: false,
      hqAutoFillPrompt: DEFAULT_HQ_GLOSSARY_PROMPT,
      entries: [],
    },
    nonTranslate: {
      enabled: false,
      entries: [],
    },
  }
}
