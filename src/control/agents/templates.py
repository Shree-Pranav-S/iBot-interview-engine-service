"""Static speech template banks for phase one.

Each bank keeps a healthy template pool to reduce repetition while template
selection stays random and checkpoint-safe.
"""

from __future__ import annotations

import random
from typing import Any

from src.core.exceptions.interview import InterviewConfigurationException

OPENING = (
    "Hello {candidate_name}, welcome to your interview with {company_name}. To begin, could you tell me about your professional background?",
    "Hi {candidate_name}, thank you for joining the {company_name} interview. Let us start with an overview of your professional background.",
    "Welcome, {candidate_name}. It is good to have you here for your interview with {company_name}. Could you begin by walking me through your professional background?",
    "Hello {candidate_name}, and welcome. Before we get into the interview, please tell me a little about your professional experience and background.",
    "Hi {candidate_name}, welcome to the interview for {company_name}. To get started, could you introduce yourself and describe your professional background?",
    "Good to meet you, {candidate_name}. Welcome to your {company_name} interview. Please begin by sharing an overview of your professional journey.",
    "Hello {candidate_name}. Thank you for being here for the {company_name} interview. Could you start by telling me about your work experience and background?",
    "Hi {candidate_name}, I am glad you could join us. Let us begin with a brief introduction to your professional background.",
    "Welcome to your interview with {company_name}, {candidate_name}. First, please walk me through your experience and the work you have done so far.",
    "Hello {candidate_name}, thanks for joining today. To open the interview, could you share your professional background and recent experience?",
)

OPENING_RESUME_AWARE = (
    "Hello {candidate_name}, welcome to your interview with {company_name}. I see you have worked with {resume_skills} — we will touch on some of that today. Could you tell me about your professional background?",
    "Hi {candidate_name}, thank you for joining the {company_name} interview. I noticed {resume_skills} on your background — we will explore a few of those areas. To begin, could you walk me through your professional experience?",
    "Welcome, {candidate_name}. It is good to have you here for your interview with {company_name}. I see experience with {resume_skills}, which aligns with what we will cover. Could you start with an overview of your background?",
    "Hello {candidate_name}, and welcome. Before we dive in, I see you have worked with {resume_skills} — we will touch on some of that today. Please tell me about your professional journey.",
    "Hi {candidate_name}, welcome to the interview for {company_name}. I see {resume_skills} in your experience, and we will cover related areas. Could you introduce yourself and describe your background?",
    "Good to meet you, {candidate_name}. Welcome to your {company_name} interview. I noticed {resume_skills} in your profile — we will explore those topics. Please share an overview of your professional background.",
    "Hello {candidate_name}. Thank you for being here for the {company_name} interview. I see you have worked with {resume_skills}, which we will discuss today. Could you tell me about your work experience?",
    "Hi {candidate_name}, I am glad you could join us. I see {resume_skills} in your background — we will touch on some of that. Let us begin with a brief introduction to your professional experience.",
    "Welcome to your interview with {company_name}, {candidate_name}. I noticed experience with {resume_skills}, and we will cover a few related areas. Please walk me through your background and recent work.",
    "Hello {candidate_name}, thanks for joining today. I see you have worked with {resume_skills} — we will explore some of that during our conversation. Could you share your professional background?",
)

NEUTRAL_FILLER = (
    "Okay.",
    "Alright.",
    "Right.",
    "Mm-hmm.",
    "Okay, let me see.",
    "Let me think about that for a second.",
    "Right, one moment.",
    "Okay, just a moment.",
    "Alright, let me see.",
    "Let me see.",
    "Understood.",
    "Got it.",
    "I see.",
    "All right, one second.",
    "Okay, I am with you.",
    "Right, I hear you.",
    "Sure, one moment.",
    "All right, give me a second.",
    "Okay, understood.",
    "Thanks, one moment.",
    "All right.",
    "Right, got it.",
    "Okay, let me check.",
    "I hear you.",
    "All right, understood.",
    "Right, I understand.",
    "Okay, right.",
    "Sure.",
    "Mm, okay.",
    "All right, let us continue.",
    "Okay, I have that.",
    "Right, thank you.",
    "Okay, just a second.",
    "All right, one moment.",
    "I understand.",
    "Right, understood.",
    "Okay then.",
    "All right then.",
    "Sure, let me think.",
    "Okay, let me process that.",
    "Right, let me process that.",
    "All right, thank you.",
    "I have that.",
    "Right, all good.",
    "Okay, all right.",
    "Understood, one second.",
    "All right, I have it.",
    "Okay, I hear you.",
    "Right, I have that.",
    "Sure, understood.",
    "Okay, noted.",
    "All right, noted.",
    "Right, noted.",
    "Okay, thank you.",
    "All right, thanks.",
    "Right, thanks.",
)

SILENCE_OFFER = (
    "Do you need some time to think?",
    "Would you like a little time to think?",
    "Do you need a moment to gather your thoughts?",
    "Would a short moment to think be helpful?",
    "Would you like a few seconds to consider your answer?",
    "Do you want a little more thinking time?",
    "Would you like me to give you a moment?",
    "Do you need a brief pause to think this through?",
    "Would you prefer a few seconds before answering?",
    "Should I give you a little time to think?",
)

THINK_WAIT = (
    "Of course. Take about fifteen seconds to think.",
    "Certainly. I will give you fifteen seconds.",
    "No problem. Take the next fifteen seconds to gather your thoughts.",
    "Sure. You have about fifteen seconds to think it through.",
    "Absolutely. Take fifteen seconds, and then we will continue.",
    "That is fine. I will give you a brief fifteen-second pause.",
    "Please take fifteen seconds to consider your response.",
    "Of course. Use the next fifteen seconds to prepare your answer.",
    "Sure, take a short fifteen-second thinking pause.",
    "Certainly. I will wait for about fifteen seconds.",
)

THINK_DECLINED = (
    "All right. Please go ahead with your answer.",
    "No problem. Please answer when you are ready.",
    "Understood. You can continue with your response now.",
    "All right, let us continue. Please share your answer.",
    "Certainly. Please go ahead and respond to the question.",
    "Okay. I am listening whenever you are ready to answer.",
    "Understood. Please continue with what you have in mind.",
    "That is fine. Go ahead with your response.",
    "All right. Please share your answer now.",
    "Okay, let us proceed with your answer.",
)

THINK_NUDGE = (
    "Let us continue. Please share your answer now.",
    "Whenever you are ready, please give me your answer.",
    "Please go ahead and share what you have.",
    "Your thinking time is up. Please respond with what you can.",
    "Let us resume. Please walk me through your answer.",
    "Please share your response now so we can continue.",
    "We can continue now. Tell me what you are thinking.",
    "Please go ahead with your best answer at this point.",
    "Let us move forward. Please give your response now.",
    "When you are ready, please answer the question.",
)

NO_RESPONSE_MOVE_ON = (
    "I did not hear a response, so I will move on to the next question.",
    "There was no response, so let us continue to the next question.",
    "I have not heard an answer, so I will move us forward.",
    "Since I did not receive a response, we will proceed to the next question.",
    "It seems there is no response for this one, so I will move on.",
    "I did not catch an answer, so let us continue with the interview.",
    "As there was no response, I will take us to the next question.",
    "We have not received an answer here, so we will move forward.",
    "I will mark this as unanswered and continue to the next question.",
    "Since there is still no response, let us move on for now.",
)

REPEAT_QUESTION = (
    "Certainly. The question is: {question}",
    "Of course. Let me repeat it: {question}",
    "Sure. I asked: {question}",
    "No problem. Here is the question again: {question}",
    "Absolutely. Please answer this question: {question}",
    "Let me say that once more: {question}",
    "Certainly. The current question was: {question}",
    "Yes, I can repeat it: {question}",
    "Here it is again: {question}",
    "Of course. I will repeat the question: {question}",
)

REPHRASE_QUESTION = (
    "Certainly. Put another way: {rephrased_question}",
    "Of course. In simpler terms: {rephrased_question}",
    "Sure. Let me phrase it differently: {rephrased_question}",
    "No problem. Here is a clearer version: {rephrased_question}",
    "Absolutely. Let me ask it another way: {rephrased_question}",
    "Let me make the wording clearer: {rephrased_question}",
    "Certainly. A simpler version is: {rephrased_question}",
    "I can rephrase that: {rephrased_question}",
    "Sure. Here is a different way to ask it: {rephrased_question}",
    "Of course. Let me ask it this way: {rephrased_question}",
)

SKIP_PARTIAL_ATTEMPT = (
    "Before we skip it, could you share whatever part of the answer you know?",
    "Please try a partial response first, even if you are not completely sure.",
    "Even a brief attempt is useful. Could you tell me what you do know?",
    "Could you give your best partial answer before we move on?",
    "That is okay, but please try the part you feel most comfortable answering.",
    "Before moving ahead, could you share any relevant idea or experience you have?",
    "Please make a short attempt, even if your answer is incomplete.",
    "Could you talk through your initial understanding before we skip the question?",
    "Let us give it one brief attempt first. What can you say about it?",
    "Please share what you can, even if you do not know the full answer.",
)

SKIP_ACKNOWLEDGEMENT = (
    "Understood. We will skip this question and move on.",
    "All right. I will mark this one as skipped and continue.",
    "That is fine. Let us move to the next question.",
    "Understood. We will leave this question here and proceed.",
    "Okay. I will skip this one and take us forward.",
    "No problem. We will move past this question now.",
    "All right, I understand. Let us continue with the interview.",
    "Understood. I will move us on from this question.",
    "Okay. We will set this question aside and proceed.",
    "That is noted. Let us go on to the next question.",
)

SKIP_RESUME_SKILL_PREFIX = (
    "It is surprising that you want to skip this question considering you mentioned {skill} in your background.",
    "I am surprised you would like to skip this, since {skill} appears in your background.",
    "Since you mentioned {skill} in your background, it is unexpected that you want to skip this question.",
    "Your background includes {skill}, so I am surprised you would prefer to skip this one.",
    "It is worth noting that {skill} is in your background, which makes this skip request unexpected.",
    "Because you listed {skill} in your background, I am surprised that you do not want to attempt this question.",
    "You mentioned experience with {skill}, so it is unexpected that you would like to pass on this question.",
    "Given that {skill} appears in your background, I am surprised you want to move past this question.",
    "I noticed {skill} in your background, so your request to skip this question is surprising.",
    "Considering your background mentions {skill}, it is unexpected that you would rather skip this question.",
)

IRRELEVANT_REDIRECT = (
    "Let us stay with the interview question. Please answer the question I asked.",
    "That does not address the current question. Please bring your response back to it.",
    "We need to keep the interview on topic. Please respond to the active question.",
    "Let us return to the question in front of us. Please share your answer.",
    "That is outside the scope of the current question. Please answer it directly.",
    "Please keep your response relevant to the interview question I asked.",
    "I cannot use that as an answer here. Please focus on the current question.",
    "Let us stay on track and continue with your answer to this question.",
    "Please bring your response back to the topic of the current interview question.",
    "That response is not relevant to what was asked. Please try the question again.",
)

ELABORATE_ANSWER = (
    "Could you elaborate a little further on that?",
    "Please add a little more detail to your answer.",
    "Could you expand on that with some reasoning or context?",
    "I would like a little more detail. Could you continue?",
    "Could you explain that more fully?",
    "Please go a bit deeper so I can understand your response.",
    "Could you add a concrete detail or example?",
    "That is quite brief. Please tell me a little more.",
    "Could you walk me through your thinking in more detail?",
    "Please expand your response enough for me to assess it.",
)

SELF_INTRO_COMPLETION_ACK = (
    "Thank you for that introduction.",
    "Thanks for walking me through your background.",
    "I appreciate the overview of your experience.",
    "Thank you for sharing your professional background.",
    "That gives me a helpful picture of your experience.",
    "I appreciate you taking the time to introduce yourself.",
    "Thanks for the context on your background and skills.",
    "That was a clear introduction. Thank you.",
    "I have a good sense of your background now. Thank you.",
    "Thanks for outlining your experience so far.",
)

SELF_INTRO_ELABORATE = (
    "Could you tell me a little more about your professional background and experience?",
    "Please expand on your recent work, core skills, and professional journey.",
    "Could you walk me through your background in a little more detail?",
    "I would like to hear more about your experience and the roles you have held.",
    "Please add some detail about your recent responsibilities and strongest skills.",
    "Could you continue with more context about your professional experience?",
    "Tell me a little more about the work you have done and your areas of expertise.",
    "Please take a bit more time to describe your background and relevant projects.",
    "Could you expand your introduction with your recent experience and key strengths?",
    "I would like a fuller picture of your background. Please continue.",
)

SUBSTANTIAL_ACKNOWLEDGEMENT = (
    "Thank you. I have noted that response, and we can move forward.",
    "Thanks for the detailed response. Let us continue.",
    "Understood. Thank you for walking me through that.",
    "Thank you. That gives me enough context to continue.",
    "I appreciate the explanation. We can move on from here.",
    "Thanks for sharing that. I have what I need for this question.",
    "That is helpful context. Thank you for the response.",
    "Thank you for explaining that. Let us proceed.",
    "I have noted your answer. Thank you.",
    "Thanks. That gives me a clear enough picture to move forward.",
    "Thank you. That was clear and useful context.",
    "Thanks for the detail. We can continue to the next point.",
    "I appreciate that explanation. Let us move ahead.",
    "That is helpful, thank you. We will continue.",
    "Thanks, that gives me what I need for this part.",
    "Understood, thank you. Let us proceed.",
    "I appreciate the context you shared. We can move on.",
    "Thank you for the thoughtful response. Let us continue.",
    "That was clear. Thank you, we can continue.",
    "Thanks, I have captured that. Let us move forward.",
    "I understand your approach. Thank you.",
    "Thank you for covering that in detail.",
    "That is useful input. Thanks, let us continue.",
    "Appreciate the detail there. We can proceed.",
    "Thanks for that context. Moving ahead.",
    "Understood. That answer is noted.",
    "I appreciate your explanation. Let us go on.",
    "Thank you, that is clear enough to proceed.",
    "Thanks for walking through that example.",
    "That helps, thank you. We will continue.",
    "I have enough detail from that response. Thank you.",
    "Thank you for the clarity there.",
    "Thanks, that is a solid explanation.",
    "I appreciate the way you explained that. Let us continue.",
    "Understood, that gives good context.",
    "Thanks for the complete response.",
    "That was helpful and specific. Thank you.",
    "I have noted that clearly. Thanks.",
    "Thanks, that answers the question well enough to proceed.",
    "I appreciate that detail. Let us move to the next area.",
    "Thank you for sharing that approach.",
    "That is clear, thank you.",
    "Thanks for that walkthrough.",
    "Understood, that is helpful.",
    "Thank you, we can continue from here.",
    "Thanks, I have captured the key points.",
    "That is good context, thank you.",
    "I appreciate the depth there. Let us proceed.",
    "Thank you for the practical explanation.",
    "Thanks, that gives me a clear view.",
    "Understood, that response is noted.",
    "Thank you for breaking that down.",
    "Thanks for the useful detail.",
    "I appreciate that answer. We can continue.",
    "Thank you, that was well explained.",
    "Thanks, this gives enough signal to move forward.",
    "Understood, thank you for the context.",
    "I have captured your response, thank you.",
    "Thanks, that is helpful for assessment.",
    "Thank you, let us continue.",
)

QUESTION_DOUBT_FALLBACK = (
    "Please interpret the question in its ordinary professional context and focus on: {question}",
    "I cannot provide the answer, but you may focus your response on this scope: {question}",
    "The intended scope is the situation described in the question: {question}",
    "Please answer based on your own experience and understanding of: {question}",
    "You can keep your response focused on the central point in this question: {question}",
    "The question is asking for your own reasoning about: {question}",
    "Please use the wording given and explain your own approach to: {question}",
    "You do not need hidden assumptions; address the question as stated: {question}",
    "Keep the answer practical and centered on: {question}",
    "Please respond at the level your experience supports, focusing on: {question}",
)

INTERVIEW_META_GUIDANCE = (
    "Explain your reasoning clearly, confidently, and thoroughly, and you will be in a strong position.",
    "Focus on showing that you understand the concepts and can explain your thinking clearly.",
    "Give direct answers, talk through your reasoning, and use concrete examples when they help.",
    "Stay calm and explain what you know clearly; thoughtful reasoning matters more than perfect wording.",
    "Answer each question directly and make your understanding visible through a clear explanation.",
    "You will be assessed on your understanding, relevance, and how clearly you explain your approach.",
    "Be concise but complete, and support your answers with reasoning or practical examples where possible.",
    "There is no special trick; listen carefully, answer honestly, and explain your thought process.",
    "Show both what you know and how you arrived at your answer, and you will do fine.",
    "Keep your answers relevant, confident, and well explained, with enough detail to show your understanding.",
)

INTERVIEW_META_TIME_REMAINING = (
    "You can view the timer at the top right of your screen.",
) * 10

QUESTION_GENERATION_FALLBACK_ACKNOWLEDGEMENT = (
    "Thank you for sharing your perspective. Let us continue.",
    "I have noted that response. Let us look at another aspect.",
    "Thank you for explaining your approach. We can continue from there.",
    "That gives me useful context. Let us move to the next point.",
    "I appreciate the detail you shared. Let us continue the discussion.",
    "Thank you for walking me through that. Let us explore another area.",
    "I have captured your response. Let us consider a different angle.",
    "Thank you for that context. We will continue with another question.",
    "I understand your perspective. Let us move forward.",
    "Thank you for describing that. Let us continue with the interview.",
    "Thanks for that context. We can continue from here.",
    "I have noted your response. Let us move to the next question.",
    "Appreciate the explanation. We will continue.",
    "Thank you for sharing that detail. Let us proceed.",
    "That was helpful context. Let us keep going.",
    "I have captured your point. Let us continue.",
    "Thanks for walking me through that. We will move ahead.",
    "Understood, thank you. Let us continue.",
    "I appreciate that response. We can move to the next one.",
    "Thank you, that is useful. Let us continue.",
    "I have what I need from that answer. Let us proceed.",
    "Thanks, we can now move to another area.",
    "That gives good context. Let us continue.",
    "Understood. We can move forward.",
    "I appreciate the detail there. Let us proceed.",
    "Thanks for explaining your thinking. Let us continue.",
    "That is clear, thank you. We will continue.",
    "I have recorded that response. Let us move ahead.",
    "Thank you for the clarification. Let us continue.",
    "Thanks, that helps. We can proceed.",
    "Understood, that is noted. Let us continue.",
    "I appreciate your perspective. Let us move forward.",
    "Thanks for sharing that example. Let us continue.",
    "That response is captured. Let us proceed.",
    "Thank you, we can move to the next question.",
    "I have noted that well. Let us continue.",
    "Thanks, let us continue with the interview.",
    "Understood. We will move to the next part.",
    "Thank you for that explanation. We can proceed.",
    "I appreciate that context. Let us continue.",
    "Thanks, this is helpful. Let us move on.",
    "Noted, thank you. We can continue.",
    "I have captured your answer. Let us proceed.",
    "Thank you for sharing your approach. Let us continue.",
    "That is useful detail. We can move ahead.",
    "Understood. Thank you, let us continue.",
    "Thanks, we will proceed to the next point.",
    "I appreciate that answer. Let us keep going.",
    "Thank you, that is clear. Let us continue.",
    "I have enough context from that. Let us proceed.",
)

TECHNICAL_QUESTION_FALLBACK = (
    "When would {concept} become a bottleneck in a production {skill} service?",
    "What failure symptom would first suggest a {concept} issue in {skill}?",
    "How would you isolate whether a bug comes from {concept} or surrounding {skill} code?",
    "Which trade-off between speed and safety matters most for {concept} in {skill}?",
    "What invariant would you enforce to prevent {concept} regressions in {skill}?",
    "How would you test {concept} behavior before shipping a {skill} change?",
    "What metric would you watch to catch early {concept} problems in {skill}?",
    "When a {skill} incident involves {concept}, what is your first diagnostic step?",
    "How would higher load change your approach to {concept} in {skill}?",
    "What design choice around {concept} most affects {skill} reliability at scale?",
)

BEHAVIOURAL_QUESTION_FALLBACK = (
    "Tell me about a specific time you demonstrated {signal} at work?",
    "Can you describe a situation where {signal} affected how you approached your work?",
    "Tell me about a difficult moment that required {signal} from you?",
    "Can you share an example of how you applied {signal} within a team?",
    "Describe a past situation where your approach to {signal} changed the outcome?",
    "Tell me about a time you had to demonstrate {signal} under pressure?",
    "Can you describe a decision that tested your {signal}?",
    "Share an example of how you developed your ability in {signal}?",
    "Tell me about a situation where a lack of {signal} created a challenge and how you responded?",
    "Can you describe a time when {signal} helped you navigate an unexpected problem?",
)

SECTION_TRANSITION = (
    "Thank you for discussing {current_section}. Let us now move to {next_section}.",
    "I appreciate your responses on {current_section}. We will continue with {next_section}.",
    "That gives us useful context on {current_section}. Let us shift to {next_section}.",
    "Thank you. We have covered {current_section}, so let us move into {next_section}.",
    "I have noted your responses for {current_section}. We can now turn to {next_section}.",
    "That completes our discussion of {current_section}. Let us continue with {next_section}.",
    "Thank you for walking through {current_section}. The next area is {next_section}.",
    "We have covered the key points for {current_section}. Let us move on to {next_section}.",
    "I appreciate that perspective on {current_section}. We will now focus on {next_section}.",
    "Thank you. Let us leave {current_section} there and continue with {next_section}.",
)

BARGE_IN_TRANSITION = (
    "As we are running short of time, we need to move on to {next_section}.",
    "I am going to pause you there because we need to move to {next_section}.",
    "To stay within the interview time, I need to move us on to {next_section}.",
    "I appreciate your thoughts. We are over time for this section, so let us continue with {next_section}.",
    "I need to step in so we can cover the remaining interview area, {next_section}.",
    "We have gone beyond the time for {current_section}, so I will move us to {next_section}.",
    "Let me pause this discussion here. We now need to continue with {next_section}.",
    "To make sure the interview stays on schedule, we must move on to {next_section}.",
    "I am going to bring this section to a close and transition to {next_section}.",
    "We are running short of time, so I will move us forward to {next_section}.",
)

BEHAVIOURAL_FORCED_TRANSITION = (
    "Due to lack of time, we need to move to the behavioural and cultural section.",
) * 10

CLOSING = (
    "Thank you for completing your interview with {company_name}. We appreciate the time and thought you shared today. Your responses have been submitted, and the recruiting team will handle the next steps.",
    "Thank you for completing your interview with {company_name}. I appreciate your time today. Your interview is now submitted, and you will hear about any next steps through the recruiting team.",
    "Thank you for completing your interview with {company_name}. That brings our conversation to a close. We appreciate your participation, and your responses have been securely submitted.",
    "Thank you for completing your interview with {company_name}. We are now at the end of the session. Your responses have been recorded for the recruiting team to review.",
    "Thank you for completing your interview with {company_name}. I appreciate you sharing your experience and perspective. The session is now complete and has been submitted.",
    "Thank you for completing your interview with {company_name}. This concludes the interview. We appreciate your participation, and the recruiting team will manage the process from here.",
    "Thank you for completing your interview with {company_name}. Your interview has now finished successfully. We appreciate the effort you put into your responses today.",
    "Thank you for completing your interview with {company_name}. We have reached the end of our time together. Your responses are submitted, and no further action is needed right now.",
    "Thank you for completing your interview with {company_name}. I appreciate your time and openness during the conversation. The interview is now complete.",
    "Thank you for completing your interview with {company_name}. That is everything for today. Your session has been submitted, and the recruiting team will take care of the next steps.",
)


_BANKS = {
    "neutral_filler": NEUTRAL_FILLER,
    "opening": OPENING,
    "opening_resume_aware": OPENING_RESUME_AWARE,
    "silence_offer": SILENCE_OFFER,
    "think_wait": THINK_WAIT,
    "think_declined": THINK_DECLINED,
    "think_nudge": THINK_NUDGE,
    "no_response_move_on": NO_RESPONSE_MOVE_ON,
    "repeat_question": REPEAT_QUESTION,
    "rephrase_question": REPHRASE_QUESTION,
    "skip_partial_attempt": SKIP_PARTIAL_ATTEMPT,
    "skip_acknowledgement": SKIP_ACKNOWLEDGEMENT,
    "skip_resume_skill_prefix": SKIP_RESUME_SKILL_PREFIX,
    "irrelevant_redirect": IRRELEVANT_REDIRECT,
    "elaborate_answer": ELABORATE_ANSWER,
    "self_intro_completion_ack": SELF_INTRO_COMPLETION_ACK,
    "self_intro_elaborate": SELF_INTRO_ELABORATE,
    "substantial_acknowledgement": SUBSTANTIAL_ACKNOWLEDGEMENT,
    "question_doubt_fallback": QUESTION_DOUBT_FALLBACK,
    "interview_meta_guidance": INTERVIEW_META_GUIDANCE,
    "interview_meta_time_remaining": INTERVIEW_META_TIME_REMAINING,
    "question_generation_fallback_acknowledgement": (
        QUESTION_GENERATION_FALLBACK_ACKNOWLEDGEMENT
    ),
    "technical_question_fallback": TECHNICAL_QUESTION_FALLBACK,
    "behavioural_question_fallback": BEHAVIOURAL_QUESTION_FALLBACK,
    "section_transition": SECTION_TRANSITION,
    "barge_in_transition": BARGE_IN_TRANSITION,
    "behavioural_forced_transition": BEHAVIOURAL_FORCED_TRANSITION,
    "closing": CLOSING,
}

if any(len(bank) < 10 for bank in _BANKS.values()):
    raise InterviewConfigurationException(
        "Every interview static template bank must have at least 10 variants"
    )


def choose_template(name: str, **values: Any) -> str:
    """
    Choose and safely format one of ten variants from a named bank at random.

    Args:
        name: The key of the template bank to pull from.
        **values: Keyword arguments to format into the string template.

    Returns:
        The fully formatted string.
    """

    bank = _BANKS[name]
    return random.SystemRandom().choice(bank).format(**values)


def choose_template_avoiding(
    name: str,
    *,
    recent: list[str] | tuple[str, ...],
    **values: Any,
) -> str:
    """
    Choose a random template variant while actively avoiding recent text and
    repetitive opening phrases (e.g. avoiding saying 'Thank you' three times in a row).

    Args:
        name: The key of the template bank.
        recent: A list of recently spoken phrases to avoid repeating.
        **values: Keyword arguments for formatting.

    Returns:
        The formatted string.
    """

    variants = list(template_variants(name, **values))
    normalized_recent = [
        " ".join(item.casefold().split()) for item in recent if str(item).strip()
    ]
    recent_openings = {" ".join(item.split()[:2]) for item in normalized_recent[-2:]}
    available = [
        item
        for item in variants
        if " ".join(item.casefold().split()) not in normalized_recent
        and " ".join(item.casefold().split()[:2]) not in recent_openings
    ]
    return random.SystemRandom().choice(available or variants)


def template_variants(name: str, **values: Any) -> tuple[str, ...]:
    """
    Return all formatted variants for deterministic fallback selection.

    Args:
        name: The key of the template bank.
        **values: Keyword arguments for formatting.

    Returns:
        A tuple containing all formatted variations.
    """

    return tuple(item.format(**values) for item in _BANKS[name])
