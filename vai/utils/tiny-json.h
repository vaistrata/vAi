/*=============================================================================
  Written by   oneway@vaistrata.com
  Streaming JSON reader, sj.h (https://github.com/rxi/sj.h) v0.4 by rxi 2025
  modified for C++. Public domain, no warranty implied.

  sj_read() returns the next token in document order and leaves values as
  {start, end} spans into the caller's buffer. Nothing is copied and no tree is
  built, so the buffer must outlive every sj_Value taken from it. Containers are
  walked with sj_iter_array / sj_iter_object, which skip whatever of the
  previous value was left unread.
=============================================================================*/

#ifndef VAI_UTILS_TINY_JSON_H
#define VAI_UTILS_TINY_JSON_H

#include <cstddef>

namespace tinyjson {


struct sj_Reader {
    char *data, *cur, *end;
    int depth;
    const char *error;
};

struct sj_Value {
    int type;
    char *start, *end;
    int depth;
};

enum { SJ_ERROR, SJ_END, SJ_ARRAY, SJ_OBJECT, SJ_NUMBER, SJ_STRING, SJ_BOOL, SJ_NULL };

inline sj_Reader sj_reader(char *data, size_t len) {
    sj_Reader r;
    r.data = data;
    r.cur = data;
    r.end = data + len;
    r.depth = 0;
    r.error = nullptr;
    return r;
}

inline bool sj__is_number_cont(char c) {
    return (c >= '0' && c <= '9')
        ||  c == 'e' || c == 'E' || c == '.' || c == '-' || c == '+';
}

inline bool sj__is_string(char *cur, char *end, const char *expect) {
    while (*expect) {
        if (cur == end || *cur != *expect)
            return false;
        expect++, cur++;
    }
    return true;
}

inline sj_Value sj_read(sj_Reader *r) {
    sj_Value res;
    res.type = SJ_ERROR;
    res.start = nullptr;
    res.end = nullptr;
    res.depth = 0;
top:
    if (r->error) {
        res.type = SJ_ERROR;
        res.start = r->cur;
        res.end = r->cur;
        return res;
    }
    if (r->cur == r->end) { r->error = "unexpected eof"; goto top; }
    res.start = r->cur;

    switch (*r->cur) {
    case ' ': case '\n': case '\r': case '\t':
    case ':': case ',':
        r->cur++;
        goto top;

    case '-': case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
        res.type = SJ_NUMBER;
        while (r->cur != r->end && sj__is_number_cont(*r->cur)) { r->cur++; }
        break;

    case '"':
        res.type = SJ_STRING;
        res.start = ++r->cur;
        for (;;) {
            if ( r->cur == r->end) { r->error = "unclosed string"; goto top; }
            if (*r->cur ==    '"') { break; }
            if (*r->cur ==   '\\') { r->cur++; }
            if ( r->cur != r->end) { r->cur++; }
        }
        res.end = r->cur++;
        return res;

    case '{': case '[':
        res.type = (*r->cur == '{') ? SJ_OBJECT : SJ_ARRAY;
        res.depth = ++r->depth;
        r->cur++;
        break;

    case '}': case ']':
        res.type = SJ_END;
        if (--r->depth < 0) {
            r->error = (*r->cur == '}') ? "stray '}'" : "stray ']'";
            goto top;
        }
        r->cur++;
        break;

    case 'n': case 't': case 'f':
        res.type = (*r->cur == 'n') ? SJ_NULL : SJ_BOOL;
        if (sj__is_string(r->cur, r->end,  "null")) { r->cur += 4; break; }
        if (sj__is_string(r->cur, r->end,  "true")) { r->cur += 4; break; }
        if (sj__is_string(r->cur, r->end, "false")) { r->cur += 5; break; }
        // fallthrough

    default:
        r->error = "unknown token";
        goto top;
    }
    res.end = r->cur;
    return res;
}

inline void sj__discard_until(sj_Reader *r, int depth) {
    sj_Value val;
    val.type = SJ_NULL;
    while (r->depth != depth && val.type != SJ_ERROR) {
        val = sj_read(r);
    }
}

inline bool sj_iter_array(sj_Reader *r, sj_Value arr, sj_Value *val) {
    sj__discard_until(r, arr.depth);
    *val = sj_read(r);
    if (val->type == SJ_ERROR || val->type == SJ_END) { return false; }
    return true;
}

inline bool sj_iter_object(sj_Reader *r, sj_Value obj, sj_Value *key, sj_Value *val) {
    sj__discard_until(r, obj.depth);
    *key = sj_read(r);
    if (key->type == SJ_ERROR || key->type == SJ_END) { return false; }
    *val = sj_read(r);
    if (val->type == SJ_END)   { r->error = "unexpected object end"; return false; }
    if (val->type == SJ_ERROR) { return false; }
    return true;
}


} // namespace tinyjson

#endif // VAI_UTILS_TINY_JSON_H
