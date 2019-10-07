"""
Pre-processor:
- remove all duplicate emojis/smileys
- normalize all positive/negative smileys into standard characters
- normalize all emojis into english descriptions of them
- tokenize the text such that emojis are a single character
"""

import emoji
import regex
import argparse
from tqdm import tqdm

def remove_duplicates(s):
    """
    Remove all duplicated emojis/smileys
    """
    out = []
    for w in s.split(" "):
        if not is_emoji(w) and not is_smiley(w):
            out.append(w)
        else:
            if out and out[-1] != w:
                out.append(w)
    return out


def is_emoji(w):
    """

    """
    w = w.strip()
    # matches all grapheme clusters that get rendered as a single character
    # allows us to detect emojis that are combinations of multiple unicode characters
    match = regex.match(r'\X', w)
    if not match or match.group() != w:
        return False
    return any(char in emoji.UNICODE_EMOJI for char in w)


def is_emoticon(w):
    """
    Returns true if w is an emoticon
    """
    if len(w) != 1:
        return False
    low = u'\U0001F600'
    high = u'\U0001F64F'
    return ord(low) <= ord(w) <= ord(high)


def is_emotionicon(w):
    """
    Returns true if w is an emotionicon (extended set of emoticons that we believe convey emotion)
    """
    w = w.strip()
    additional = {
        u':thumbs_up:': u'\U0001F44D',
        u':thumbs_down:': u'\U0001F44E',
        u':broken_heart:': u'\U0001F494',
        u':beating_heart:': u'\U0001F493',
        u':red_heart:': u'\U00002764',
        u':two_hearts:': u'\U0001F495',
        u':hearts:': u'\U00002665',
    }
    reverse_additional = {additional[k]: k for k in additional}
    if len(w) != 1:
        return False
    return is_emoticon(w) or w in reverse_additional


def is_positive_emotionicon(w):
    """
    Returns true if this is a positive emotionicon
    We have hardcoded these for now.
    """
    positives = {
        # emotionicons
        u':thumbs_up:': u'\U0001F44D',
        u':beating_heart:': u'\U0001F493',
        u':red_heart:': u'\U00002764',
        u':two_hearts:': u'\U0001F495',
        u':hearts:': u'\U00002665',
        # standard emoticons
        u':beaming_face_with_smiling_eyes:': u'\U0001F601',
        u':face_with_tears_of_joy:': u'\U0001F602',
        u':grinning_face:': u'\U0001F600',
        u':grinning_face_with_big_eyes:': u'\U0001F603',
        u':grinning_face_with_smiling_eyes:': u'\U0001F604',
        u':grinning_face_with_sweat:': u'\U0001F605',
        u':grinning_squinting_face:': u'\U0001F606',
        u':smiling_face_with_halo:': u'\U0001F607',
        u':winking_face:': u'\U0001F609',
        u':smiling_face_with_smiling_eyes:': u'\U0001F60A',
        u':yum:': u'\U0001F60B',
        u':relieved_face:': u'\U0001F60C',
        u':smiling_face_with_heart-eyes:': u'\U0001F60D',
        u':smiling_face_with_sunglasses:': u'\U0001F60E',
        u':smirking_face:': u'\U0001F60F',
        u':kissing:': u'\U0001F617',
        u':face_blowing_a_kiss:': u'\U0001F618',
        u':kissing_face_with_smiling_eyes:': u'\U0001F619',
        u':kissing_closed_eyes:': u'\U0001F61A',
        u':stuck_out_tongue:': u'\U0001F61B',
        u':stuck_out_tongue_closed_eyes:': u'\U0001F61D',
        u':stuck_out_tongue_winking_eye:': u'\U0001F61C',
        u':grinning_cat_face_with_smiling_eyes:': u'\U0001F638',
        u':joy_cat:': u'\U0001F639',
        u':smiley_cat:': u'\U0001F63A',
        u':smiling_cat_face_with_heart-eyes:': u'\U0001F63B',
        u':smirk_cat:': u'\U0001F63C',
        u':kissing_cat:': u'\U0001F63D',
        u':slightly_smiling_face:': u'\U0001F642',
        u':upside__down_face:': u'\U0001F643',
    }
    positives_reversed = {positives[k]: k for k in positives}
    return w.strip() in positives_reversed


def is_negative_emotionicon(w):
    """
    Returns true if this is a negative emotionicon; these are hardcoded
    """
    negatives = {
        # emotionicons
        u':thumbs_down:': u'\U0001F44E',
        u':broken_heart:': u'\U0001F494',
        # standard emoticons
        u':neutral_face:': u'\U0001F610',
        u':expressionless:': u'\U0001F611',
        u':unamused_face:': u'\U0001F612',
        u':sweat:': u'\U0001F613',
        u':pensive:': u'\U0001F614',
        u':confused_face:': u'\U0001F615',
        u':confounded_face:': u'\U0001F616',
        u':disappointed:': u'\U0001F61E',
        u':worried_face:': u'\U0001F61F',
        u':angry:': u'\U0001F620',
        u':rage:': u'\U0001F621',
        u':cry:': u'\U0001F622',
        u':persevere:': u'\U0001F623',
        u':triumph:': u'\U0001F624',
        u':disappointed_relieved:': u'\U0001F625',
        u':frowning:': u'\U0001F626',
        u':anguished:': u'\U0001F627',
        u':fearful:': u'\U0001F628',
        u':weary:': u'\U0001F629',
        u':sleepy:': u'\U0001F62A',
        u':tired_face:': u'\U0001F62B',
        u':grimacing:': u'\U0001F62C',
        u':sob:': u'\U0001F62D',
        u':face_with_open_mouth:': u'\U0001F62E',
        u':hushed:': u'\U0001F62F',
        u':anxious_face_with_sweat:': u'\U0001F630',
        u':face_screaming_in_fear:': u'\U0001F631',
        u':pouting_cat:': u'\U0001F63E',
        u':crying_cat_face:': u'\U0001F63F',
        u':slightly_frowning_face:': u'\U0001F641',
    }
    negatives_reversed = {negatives[k]: k for k in negatives}
    return w.strip() in negatives_reversed


def emoji_to_english(w):
    """
    Converts an emoji into an english description of it
    """
    return emoji.UNICODE_EMOJI[w.strip()]


def emoji_to_ascii(w):
    """
    Converts some emojis into their ASCII representations, courtesy of
    https://stackoverflow.com/a/29581503
    Returns None if no representation found
    TODO: need to include representations for more characters!!!
    """
    representations = {
        128075: 'o/',
        128148: '</3',
        128151: '<3',
        # start of emoticons
        128513: 'xD',
        128514: ":')",
        128515: ':-))',
        128516: ':>',
        128519: 'O:-)',
        128520: '>;)',
        128521: ';^)',
        128528: ':|',
        128530: '>:[',
        128534: '%-)',
        128540: '>:P',
        128544: '>:(',
        128545: '>:\\\\',
        128546: ":'(",
        128548: '>_>^',
        128555: '|;-)',
        128560: ':-###..',
        128561: 'v.v',
        128562: '>:O',
        128563: ':$',
        128565: '#-)',
        128566: ':X',
        128572: ':-J',
        128573: ':^*',
        128581: 'ಠ_ಠ',
        128591: "",
        # end of unicode encodings
        28582: '\\\\o/'
    }
    return representations[ord(w)] if ord(w) in representations else None


def is_smiley(w: str):
    """
    Returns true if the given word is an ASCII smiley
    """
    w = w.strip()
    forward = "[<>]?[BXx:;*=8][']?[\-o\*\'^\-‑]?[\(\)\[\]cOo0*xX#bDPdp><|@\\\/\{\}3]+"
    reverse = "[\(\)\[\]cOo0*xX#bDPdp><|@\\\/\{\}3]+[\-o\*\'^\-‑]?[']?[BXx:;=8][<>]"
    eastern = "[\(\{\[]?[=\\\/*#]?[;-<>\-ovO0T'Q\+^ー.°~＾*][_.o0Onv·J][;-<>\-ovO0T'Q\+^ー.°~＾*][=\\\/*#]?[\)\}\]]?"
    hearts = "<+[\\/]*3+"
    pattern = regex.compile("(?:{}|{}]|{}|{})".format(forward, reverse, eastern, hearts))
    m = pattern.match(w)
    return m is not None and m.group() is w


def preprocess(s: str):
    """
    - Remove all non-emotionicon emojis (TODO: should we convert them into english equivalents?)
    - Convert each emotionicon emoji into a corresponding happy/sad smiley (TODO: should we convert into more smileys instead)
    - Currently do not normalize the smileys (TODO: should be normalize?)
    """
    out = []

    # splits by grapheme characters
    split = regex.findall(r'\X', s)

    for i, token in enumerate(split):
        # check if token is emoji
        if token in emoji.UNICODE_EMOJI:
            # do not add duplicated emojis
            if not i or token != split[i-1]:
                if is_positive_emotionicon(token):
                    out.append(" :) ")
                elif is_negative_emotionicon(token):
                    out.append(" :( ")
        # if not emoji always keep it
        else:
            out.append(token)

    return "".join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="path to data file to process")
    parser.add_argument("out", help="path to output file")

    args = parser.parse_args()

    with open(args.data, "r") as fdata:
        with open(args.out, "w") as fout:
            for line in tqdm(fdata):
                split = line.split("\t")
                out = [""]*len(split)
                for i, x in enumerate(split):
                    out[i] = x
                for i in range(1, 4):
                    out[i] = preprocess(split[i])
                fout.write("\t".join(out))
