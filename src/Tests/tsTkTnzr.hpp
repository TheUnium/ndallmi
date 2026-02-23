// Created by Unium on 23.02.26

#pragma once

#include "../Tokenizer/tkTkBpe_.hpp"
#include "tsTsTstf.hpp"

#include <string>
#include <vector>

using namespace TK;

// <<<s_start(tokenizerc)
// --- tokenizer construction
TEST(construct_default) {
    CBpeTokenizer tok;
    Check(tok.iVocabSize() == 0, "default vocab should be empty");
}

TEST(construct_special_ids_default) {
    CBpeTokenizer tok;
    Check(tok.iBosId() == 0, "default bos should be 0");
    Check(tok.iEosId() == 0, "default eos should be 0");
    Check(tok.iUnkId() == 0, "default unk should be 0");
}

TEST(lookup_empty_vocab) {
    CBpeTokenizer tok;
    Check(tok.iLookup("anything") == -1, "empty vocab should return -1");
}

TEST(encode_empty_string) {
    CBpeTokenizer tok;
    auto vi = tok.Encode("");
    Check(vi.size() == 0, "empty string should produce no tokens");
}

TEST(decode_empty_tokens) {
    CBpeTokenizer tok;
    auto sz = tok.Decode({});
    Check(sz.empty(), "empty tokens should decode to empty string");
}
// >>>s_end(tokenizerc)

// <<<s_start(tmv)
// --- manual vocab building
TEST(add_token_basic) {
    CBpeTokenizer tok;
    tok.AddToken("hello", 0, 0.0f);
    Check(tok.iVocabSize() == 1, "vocab size should be 1");
    Check(tok.iLookup("hello") == 0, "hello should be at id 0");
}

TEST(add_token_sparse_ids) {
    CBpeTokenizer tok;
    tok.AddToken("a", 0);
    tok.AddToken("b", 100);
    Check(tok.iVocabSize() == 101, "vocab should expand to fit id 100");
    Check(tok.iLookup("a") == 0, "a at 0");
    Check(tok.iLookup("b") == 100, "b at 100");
}

TEST(add_special_token) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<pad>", 0);
    Check(tok.iLookup("<pad>") == 0, "pad lookup");
    auto sz = tok.Decode({0});
    Check(sz.empty(), "special token should not appear in decode");
}

TEST(add_multiple_tokens) {
    CBpeTokenizer tok;
    tok.AddToken("a", 0);
    tok.AddToken("b", 1);
    tok.AddToken("c", 2);
    Check(tok.iVocabSize() == 3, "should have 3 tokens");
    Check(tok.iLookup("a") == 0, "a");
    Check(tok.iLookup("b") == 1, "b");
    Check(tok.iLookup("c") == 2, "c");
}

TEST(add_token_overwrite) {
    CBpeTokenizer tok;
    tok.AddToken("old", 0);
    tok.AddToken("new", 0);
    Check(tok.iLookup("new") == 0, "new should replace old at id 0");
}
// >>>s_end(tmv)

// <<<s_start(tenm)
// --- encoding without merges
TEST(encode_single_char) {
    CBpeTokenizer tok;
    tok.AddToken("a", 0);
    tok.BuildFromVocab();
    auto vi = tok.Encode("a");
    Check(vi.size() == 1, "single char should be 1 token");
    Check(vi[0] == 0, "should be id 0");
}

TEST(encode_three_chars_no_merge) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1);
    tok.AddToken("b", 2);
    tok.AddToken("c", 3);
    tok.BuildFromVocab();
    auto vi = tok.Encode("abc");
    Check(vi.size() == 3, "abc should be 3 tokens");
    Check(vi[0] == 1, "first is a");
    Check(vi[1] == 2, "second is b");
    Check(vi[2] == 3, "third is c");
}

TEST(encode_unknown_char_falls_back) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1);
    tok.BuildFromVocab();
    auto vi = tok.Encode("z");
    Check(vi.size() == 1, "should produce 1 token");
    Check(vi[0] == 0, "should be unk id");
}

TEST(encode_mixed_known_unknown) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1);
    tok.AddToken("c", 2);
    tok.BuildFromVocab();
    auto vi = tok.Encode("abc");
    Check(vi.size() == 3, "should be 3 tokens");
    Check(vi[0] == 1, "a");
    Check(vi[1] == 0, "b -> unk");
    Check(vi[2] == 2, "c");
}
// >>>s_end(tenm)

// <<<s_start(tem)
// --- encoding with merges
TEST(encode_simple_merge) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("ab", 3, 10.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("ab");
    Check(vi.size() == 1, "ab should merge to 1 token");
    Check(vi[0] == 3, "should be token id 3");
}

TEST(encode_merge_left_unmerged) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("c", 3, 0.0f);
    tok.AddToken("ab", 4, 10.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("abc");
    Check(vi.size() == 2, "abc should be ab + c");
    Check(vi[0] == 4, "first is ab");
    Check(vi[1] == 3, "second is c");
}

TEST(encode_merge_right_unmerged) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("c", 3, 0.0f);
    tok.AddToken("bc", 4, 10.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("abc");
    Check(vi.size() == 2, "abc should be a + bc");
    Check(vi[0] == 1, "first is a");
    Check(vi[1] == 4, "second is bc");
}

TEST(encode_cascading_merges) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("c", 3, 0.0f);
    tok.AddToken("d", 4, 0.0f);
    tok.AddToken("ab", 5, 100.0f);
    tok.AddToken("cd", 6, 99.0f);
    tok.AddToken("abcd", 7, 50.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("abcd");
    Check(vi.size() == 1, "abcd should fully merge");
    Check(vi[0] == 7, "should be abcd token");
}

TEST(encode_merge_priority) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("c", 3, 0.0f);
    tok.AddToken("ab", 4, 100.0f);
    tok.AddToken("bc", 5, 50.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("abc");
    Check(vi.size() == 2, "abc should be ab + c");
    Check(vi[0] == 4, "first is ab (higher priority)");
    Check(vi[1] == 3, "second is c");
}

TEST(encode_repeated_pair) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("aa", 2, 10.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("aa");
    Check(vi.size() == 1, "aa should merge");
    Check(vi[0] == 2, "should be aa token");
}

TEST(encode_three_same_chars) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("aa", 2, 10.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("aaa");
    Check(vi.size() == 2, "aaa should be aa + a");
}

TEST(encode_no_merge_without_built) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("ab", 3, 10.0f);
    auto vi = tok.Encode("ab");
    Check(vi.size() == 2, "without build, no merges should happen");
}
// >>>s_end(tem)

// <<<s_start(tokenizerd)
// --- decoding
TEST(decode_single_token) {
    CBpeTokenizer tok;
    tok.AddToken("hello", 0);
    auto sz = tok.Decode({0});
    Check(sz == "hello", "should decode to hello");
}

TEST(decode_multiple_tokens) {
    CBpeTokenizer tok;
    tok.AddToken("hel", 0);
    tok.AddToken("lo", 1);
    auto sz = tok.Decode({0, 1});
    Check(sz == "hello", "should concatenate to hello");
}

TEST(decode_skips_special) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<s>", 0);
    tok.AddSpecialToken("</s>", 1);
    tok.AddToken("hi", 2);
    auto sz = tok.Decode({0, 2, 1});
    Check(sz == "hi", "special tokens should be skipped");
}

TEST(decode_skips_negative_id) {
    CBpeTokenizer tok;
    tok.AddToken("a", 0);
    auto sz = tok.Decode({-1, 0, -5});
    Check(sz == "a", "negative IDs should be skipped");
}

TEST(decode_skips_out_of_range) {
    CBpeTokenizer tok;
    tok.AddToken("a", 0);
    auto sz = tok.Decode({999999, 0});
    Check(sz == "a", "out-of-range IDs should be skipped");
}

TEST(decode_token_valid) {
    CBpeTokenizer tok;
    tok.AddToken("world", 0);
    auto sz = tok.DecodeToken(0);
    Check(sz == "world", "DecodeToken should return world");
}

TEST(decode_token_special_returns_text) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<bos>", 0);
    auto sz = tok.DecodeToken(0);
    Check(sz == "<bos>", "DecodeToken of special should return its text");
}

TEST(decode_token_invalid_id) {
    CBpeTokenizer tok;
    auto sz = tok.DecodeToken(-1);
    Check(sz == "<INVALID>", "invalid id should return <INVALID>");
}

TEST(decode_token_out_of_range) {
    CBpeTokenizer tok;
    tok.AddToken("a", 0);
    auto sz = tok.DecodeToken(999);
    Check(sz == "<INVALID>", "out-of-range should return <INVALID>");
}
// >>>s_end(tokenizerd)

// <<<s_start(tokenizerr)
// --- encode/decode roundtrip
TEST(roundtrip_single_char) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1);
    tok.BuildFromVocab();
    auto vi = tok.Encode("a");
    auto sz = tok.Decode(vi);
    Check(sz == "a", "roundtrip a");
}

TEST(roundtrip_no_merges) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("x", 1);
    tok.AddToken("y", 2);
    tok.AddToken("z", 3);
    tok.BuildFromVocab();
    auto vi = tok.Encode("xyz");
    auto sz = tok.Decode(vi);
    Check(sz == "xyz", "roundtrip xyz");
}

TEST(roundtrip_with_merges) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("h", 1, 0.0f);
    tok.AddToken("e", 2, 0.0f);
    tok.AddToken("l", 3, 0.0f);
    tok.AddToken("o", 4, 0.0f);
    tok.AddToken("he", 5, 100.0f);
    tok.AddToken("ll", 6, 99.0f);
    tok.AddToken("hell", 7, 50.0f);
    tok.AddToken("hello", 8, 40.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("hello");
    auto sz = tok.Decode(vi);
    Check(sz == "hello", "roundtrip hello");
    Check(vi.size() == 1, "should be fully merged");
}

TEST(roundtrip_repeated) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("ab", 3, 10.0f);
    tok.BuildFromVocab();
    std::string szIn = "ababab";
    auto vi = tok.Encode(szIn);
    auto sz = tok.Decode(vi);
    Check(sz == szIn, "roundtrip ababab");
    Check(vi.size() == 3, "should be 3 ab tokens");
}

TEST(roundtrip_long_string) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("ab", 3, 10.0f);
    tok.BuildFromVocab();
    std::string szIn;
    for (int i = 0; i < 200; i++)
        szIn += "ab";
    auto vi = tok.Encode(szIn);
    auto sz = tok.Decode(vi);
    Check(sz == szIn, "roundtrip long string");
}
// >>>s_end(tokenizerr)

// <<<s_start(tokenizer_bos_eos)
// --- bos/eos wrapping
TEST(encode_with_bos) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddSpecialToken("<s>", 1);
    tok.AddToken("a", 2);
    tok.BuildFromVocab();
    auto vi = tok.EncodeWithBos("a");
    Check(vi.size() == 2, "should have bos + token");
    Check(vi[0] == tok.iBosId(), "first should be bos");
    Check(vi[1] == 2, "second should be a");
}

TEST(encode_with_bos_eos) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddSpecialToken("<s>", 1);
    tok.AddSpecialToken("</s>", 2);
    tok.AddToken("a", 3);
    tok.BuildFromVocab();
    auto vi = tok.EncodeWithBosEos("a");
    Check(vi.size() == 3, "should have bos + token + eos");
    Check(vi[0] == tok.iBosId(), "first is bos");
    Check(vi[2] == tok.iEosId(), "last is eos");
}

TEST(encode_with_bos_empty) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddSpecialToken("<s>", 1);
    tok.BuildFromVocab();
    auto vi = tok.EncodeWithBos("");
    Check(vi.size() == 1, "empty should just have bos");
    Check(vi[0] == tok.iBosId(), "should be bos");
}

TEST(encode_with_bos_eos_empty) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddSpecialToken("<s>", 1);
    tok.AddSpecialToken("</s>", 2);
    tok.BuildFromVocab();
    auto vi = tok.EncodeWithBosEos("");
    Check(vi.size() == 2, "empty should have bos + eos");
    Check(vi[0] == tok.iBosId(), "first is bos");
    Check(vi[1] == tok.iEosId(), "second is eos");
}

TEST(decode_strips_bos_eos) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("h", 1);
    tok.AddToken("i", 2);
    tok.AddToken("hi", 3, 10.0f);
    tok.BuildFromVocab();
    auto vi = tok.EncodeWithBosEos("hi");
    auto sz = tok.Decode(vi);
    Check(sz == "hi", "bos/eos should be stripped in decode");
}
// >>>s_end(tokenizer_bos_eos)

// <<<s_start(tokenizera)
// --- added token handling
TEST(added_token_encode_match) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1);
    tok.AddToken("b", 2);
    tok.AddSpecialToken("<special>", 3);
    tok.BuildFromVocab();
    auto vi = tok.Encode("a<special>b");
    Check(vi.size() == 3, "should be a + <special> + b");
    Check(vi[0] == 1, "first is a");
    Check(vi[1] == 3, "second is <special>");
    Check(vi[2] == 2, "third is b");
}

TEST(added_token_decode_special_skipped) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("x", 1);
    tok.AddSpecialToken("<sep>", 2);
    auto sz = tok.Decode({1, 2, 1});
    Check(sz == "xx", "special added token should be skipped in decode");
}

TEST(added_token_multiple_specials) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddSpecialToken("<a>", 1);
    tok.AddSpecialToken("<bb>", 2);
    tok.AddToken("x", 3);
    tok.BuildFromVocab();
    auto vi = tok.Encode("<bb>x<a>");
    Check(vi.size() == 3, "should match both specials");
    Check(vi[0] == 2, "<bb>");
    Check(vi[1] == 3, "x");
    Check(vi[2] == 1, "<a>");
}
// >>>s_end(tokenizera)

// <<<s_start(tokenizer_edge)
// --- random edge cases n shit
TEST(encode_single_unknown) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.BuildFromVocab();
    auto vi = tok.Encode("z");
    Check(vi.size() == 1, "should produce 1 unk token");
    Check(vi[0] == 0, "should be unk");
}

TEST(encode_all_unknown) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.BuildFromVocab();
    auto vi = tok.Encode("xyz");
    Check(vi.size() == 3, "each char should be unk");
    for (size_t i = 0; i < vi.size(); i++)
        Check(vi[i] == 0, "should be unk");
}

TEST(vocab_size_grows) {
    CBpeTokenizer tok;
    Check(tok.iVocabSize() == 0, "starts at 0");
    tok.AddToken("a", 5);
    Check(tok.iVocabSize() == 6, "should be 6 after adding id 5");
    tok.AddToken("b", 3);
    Check(tok.iVocabSize() == 6, "should still be 6");
    tok.AddToken("c", 10);
    Check(tok.iVocabSize() == 11, "should be 11 after adding id 10");
}

TEST(merge_does_not_affect_unrelated) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("c", 3, 0.0f);
    tok.AddToken("ab", 4, 10.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("c");
    Check(vi.size() == 1, "c should be 1 token");
    Check(vi[0] == 3, "should be c");
}

TEST(multiple_merges_same_string) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    tok.AddToken("a", 1, 0.0f);
    tok.AddToken("b", 2, 0.0f);
    tok.AddToken("c", 3, 0.0f);
    tok.AddToken("bc", 4, 100.0f);
    tok.AddToken("abc", 5, 50.0f);
    tok.BuildFromVocab();
    auto vi = tok.Encode("abc");
    Check(vi.size() == 1, "abc should fully merge");
    Check(vi[0] == 5, "should be abc token");
}

TEST(decode_only_specials) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<a>", 0);
    tok.AddSpecialToken("<b>", 1);
    tok.AddSpecialToken("<c>", 2);
    auto sz = tok.Decode({0, 1, 2});
    Check(sz.empty(), "all special should produce empty decode");
}

TEST(decode_preserves_order) {
    CBpeTokenizer tok;
    tok.AddToken("x", 0);
    tok.AddToken("y", 1);
    tok.AddToken("z", 2);
    auto sz = tok.Decode({2, 0, 1});
    Check(sz == "zxy", "decode should preserve token order");
}
// >>>s_end(tokenizer_edge)

// <<<s_start(tokenizer_bench)
// --- tokenizer benchmarks
TEST(bench_encode_short) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    for (int i = 0; i < 26; i++)
        tok.AddToken(std::string(1, 'a' + i), i + 1, 0.0f);
    tok.AddToken("th", 27, 100.0f);
    tok.AddToken("he", 28, 99.0f);
    tok.AddToken("the", 29, 50.0f);
    tok.BuildFromVocab();
    std::string szIn = "the quick brown fox";
    Bench("encode 19 chars", 10000, [&]() {
        auto vi = tok.Encode(szIn);
        (void)vi;
    });
}

TEST(bench_encode_medium) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    for (int i = 0; i < 26; i++)
        tok.AddToken(std::string(1, 'a' + i), i + 1, 0.0f);
    tok.AddToken(" ", 27, 0.0f);
    tok.AddToken("th", 28, 100.0f);
    tok.AddToken("he", 29, 99.0f);
    tok.AddToken("in", 30, 98.0f);
    tok.AddToken("the", 31, 50.0f);
    tok.BuildFromVocab();
    std::string szIn;
    for (int i = 0; i < 50; i++)
        szIn += "the quick brown fox ";
    Bench("encode 1000 chars", 1000, [&]() {
        auto vi = tok.Encode(szIn);
        (void)vi;
    });
}

TEST(bench_encode_long) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    for (int i = 0; i < 26; i++)
        tok.AddToken(std::string(1, 'a' + i), i + 1, 0.0f);
    tok.AddToken(" ", 27, 0.0f);
    tok.AddToken("ab", 28, 100.0f);
    tok.AddToken("cd", 29, 99.0f);
    tok.AddToken("abcd", 30, 50.0f);
    tok.BuildFromVocab();
    std::string szIn;
    for (int i = 0; i < 1000; i++)
        szIn += "abcd ";
    Bench("encode 5000 chars", 200, [&]() {
        auto vi = tok.Encode(szIn);
        (void)vi;
    });
}

TEST(bench_decode_short) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    for (int i = 0; i < 26; i++)
        tok.AddToken(std::string(1, 'a' + i), i + 1, 0.0f);
    tok.BuildFromVocab();
    std::vector<int32_t> viTokens;
    for (int i = 0; i < 20; i++)
        viTokens.push_back((i % 26) + 1);
    Bench("decode 20 tokens", 50000, [&]() {
        auto sz = tok.Decode(viTokens);
        (void)sz;
    });
}

TEST(bench_decode_long) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    for (int i = 0; i < 26; i++)
        tok.AddToken(std::string(1, 'a' + i), i + 1, 0.0f);
    tok.AddToken("ab", 27);
    tok.BuildFromVocab();
    std::vector<int32_t> viTokens;
    for (int i = 0; i < 2000; i++)
        viTokens.push_back((i % 27) + 1);
    Bench("decode 2000 tokens", 5000, [&]() {
        auto sz = tok.Decode(viTokens);
        (void)sz;
    });
}

TEST(bench_roundtrip) {
    CBpeTokenizer tok;
    tok.AddSpecialToken("<unk>", 0);
    for (int i = 0; i < 26; i++)
        tok.AddToken(std::string(1, 'a' + i), i + 1, 0.0f);
    tok.AddToken(" ", 27, 0.0f);
    tok.AddToken("th", 28, 100.0f);
    tok.AddToken("he", 29, 99.0f);
    tok.AddToken("the", 30, 50.0f);
    tok.BuildFromVocab();
    std::string szIn;
    for (int i = 0; i < 100; i++)
        szIn += "the quick brown fox ";
    Bench("roundtrip 2000 chars", 500, [&]() {
        auto vi = tok.Encode(szIn);
        auto sz = tok.Decode(vi);
        (void)sz;
    });
}
// >>>s_end(tokenizer_bench)
