module evaluate;

import std.stdio;
import std.json;
import std.array;
import std.algorithm;
import std.string;
import std.conv;
import std.file;
import std.math;

string strRepeat(string s, size_t n) {
    string result;
    foreach (_; 0 .. n)
        result ~= s;
    return result;
}

// ============================================================
// DATA MODELS
// ============================================================

struct GroundTruthSample {
    string sampleId;
    string imagePath;
    string[] objectsPresent;
    string expectedDecision;
    string[] applicableRules;
    string notes;
}

struct LLMResponse {
    string sampleId;
    string rawOutput;
    string[] identifiedObjects;
    string decision;
    string reasoning;
}

struct SampleScore {
    string sampleId;
    double perceptionIou;
    bool strategyCorrect;
    double combinedScore;
    string[] gtObjects;
    string[] predObjects;
    string gtDecision;
    string predDecision;
    string[] gtRules;
    string errorType;
}

// ============================================================
// HELPERS
// ============================================================

string normalize(string text) {
    return text.strip().toLower().replace("_", " ").replace("-", " ");
}

string[string] toSet(string[] arr) {
    string[string] s;
    foreach (e; arr)
        s[e] = e;
    return s;
}

double jaccard(string[] a, string[] b) {
    auto setA = toSet(a);
    auto setB = toSet(b);

    if (setA.empty && setB.empty)
        return 1.0;
    if (setA.empty || setB.empty)
        return 0.0;

    size_t intersection = 0;
    foreach (key; setA.keys) {
        if (key in setB)
            intersection++;
    }
    size_t unionSize = setA.length + setB.length - intersection;
    return cast(double) intersection / cast(double) unionSize;
}

string[] getAllKeys(string[] a, string[] b) {
    auto s = toSet(a);
    foreach (e; b)
        s[e] = e;
    return s.keys.array;
}

string[] sortedDup(string[] arr) {
    auto d = arr.dup;
    d.sort!((a, b) => a < b);
    return d;
}

// ============================================================
// METRIC COMPUTATION
// ============================================================

struct PerceptionMetrics {
    double precision;
    double recall;
    double f1;
    double exactMatchRate;
    string[] objectTypes;
    double[] objectPrecisions;
    double[] objectRecalls;
}

PerceptionMetrics computePerceptionMetrics(GroundTruthSample[] samples, LLMResponse[] responses) {
    int tp = 0, fp = 0, fn = 0;
    int[string] typeTp;
    int[string] typeFp;
    int[string] typeFn;

    foreach (i, sample; samples) {
        auto resp = responses[i];
        auto gtSet = toSet(sample.objectsPresent);
        auto predSet = toSet(resp.identifiedObjects);
        string[] allObjs = getAllKeys(sample.objectsPresent, resp.identifiedObjects);

        foreach (obj; allObjs) {
            bool inGt = (obj in gtSet) !is null;
            bool inPred = (obj in predSet) !is null;

            if (inGt && inPred) {
                tp++;
                typeTp[obj] = (obj in typeTp) ? typeTp[obj] + 1 : 1;
            } else if (inPred && !inGt) {
                fp++;
                typeFp[obj] = (obj in typeFp) ? typeFp[obj] + 1 : 1;
            } else if (inGt && !inPred) {
                fn++;
                typeFn[obj] = (obj in typeFn) ? typeFn[obj] + 1 : 1;
            }
        }
    }

    double precision = (tp + fp) > 0 ? cast(double) tp / (tp + fp) : 0.0;
    double recall = (tp + fn) > 0 ? cast(double) tp / (tp + fn) : 0.0;
    double f1 = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;

    int exactMatches = 0;
    foreach (i, sample; samples) {
        auto resp = responses[i];
        auto gs = toSet(sample.objectsPresent);
        auto ps = toSet(resp.identifiedObjects);
        if (gs.length == ps.length) {
            bool match = true;
            foreach (e; gs.keys) {
                if ((e in ps) is null) {
                    match = false;
                    break;
                }
            }
            if (match)
                exactMatches++;
        }
    }

    auto allTypes = typeTp.keys.array.sortedDup;
    double[] objPrec;
    double[] objRec;
    foreach (obj; allTypes) {
        int t = (obj in typeTp) ? typeTp[obj] : 0;
        int f = (obj in typeFp) ? typeFp[obj] : 0;
        int n = (obj in typeFn) ? typeFn[obj] : 0;
        objPrec ~= (t + f) > 0 ? cast(double) t / (t + f) : 0.0;
        objRec ~= (t + n) > 0 ? cast(double) t / (t + n) : 0.0;
    }

    return PerceptionMetrics(precision, recall, f1, samples.length > 0
            ? cast(double) exactMatches / samples.length : 0.0, allTypes, objPrec, objRec);
}

struct StrategyMetrics {
    double decisionAccuracy;
    double ruleRecall;
    double rulePrecision;
    int hallucinatedRules;
    int missedRules;
}

StrategyMetrics computeStrategyMetrics(GroundTruthSample[] samples, LLMResponse[] responses) {
    int correct = 0;
    int ruleTp = 0, ruleFp = 0, ruleFn = 0;

    foreach (i, sample; samples) {
        auto resp = responses[i];

        if (normalize(sample.expectedDecision) == normalize(resp.decision))
            correct++;

        string[string] gtRules;
        foreach (r; sample.applicableRules)
            gtRules[normalize(r)] = normalize(r);

        string combined = resp.rawOutput ~ " " ~ resp.reasoning;
        string combinedNorm = normalize(combined);

        foreach (rule; gtRules.keys) {
            if (combinedNorm.canFind(rule))
                ruleTp++;
            else
                ruleFn++;
        }
    }

    size_t total = samples.length;
    double ruleRec = (ruleTp + ruleFn) > 0 ? cast(double) ruleTp / (ruleTp + ruleFn) : 0.0;
    double rulePrec = (ruleTp + ruleFp) > 0 ? cast(double) ruleTp / (ruleTp + ruleFp) : 0.0;

    return StrategyMetrics(total > 0 ? cast(double) correct / total : 0.0, ruleRec,
            rulePrec, ruleFp, ruleFn);
}

SampleScore[] scoreSamples(GroundTruthSample[] samples, LLMResponse[] responses,
        double percWeight = 0.4, double stratWeight = 0.6) {
    SampleScore[] scores;

    foreach (i, sample; samples) {
        auto resp = responses[i];
        double iou = jaccard(sample.objectsPresent, resp.identifiedObjects);
        bool correct = normalize(sample.expectedDecision) == normalize(resp.decision);

        double percFactor = iou;
        double stratFactor = correct ? 1.0 : 0.0;
        double combined = pow(percFactor, percWeight) * pow(stratFactor, stratWeight);

        string errorType;
        bool percFail = iou < 0.5;
        bool stratFail = !correct;
        if (percFail && stratFail)
            errorType = "both";
        else if (percFail)
            errorType = "perception_fail";
        else if (stratFail)
            errorType = "strategy_fail";
        else
            errorType = "none";

        scores ~= SampleScore(sample.sampleId, iou, correct, combined, sortedDup(sample.objectsPresent),
                sortedDup(resp.identifiedObjects), sample.expectedDecision,
                resp.decision, sample.applicableRules.dup, errorType);
    }
    return scores;
}

// ============================================================
// FILE I/O
// ============================================================

GroundTruthSample[] loadDataset(string path) {
    string text = readText(path);
    auto json = parseJSON(text);
    auto arr = json.array;
    GroundTruthSample[] samples;

    foreach (item; arr) {
        string[] objs;
        foreach (o; item["objects_present"].array)
            objs ~= o.toString();
        string[] rules;
        foreach (r; item["applicable_rules"].array)
            rules ~= r.toString();

        samples ~= GroundTruthSample(item["sample_id"].toString(),
                item["image_path"].toString(), objs,
                item["expected_decision"].toString(), rules, item["notes"].toString());
    }
    return samples;
}

LLMResponse[] loadResponses(string path) {
    string text = readText(path);
    auto json = parseJSON(text);
    auto arr = json.array;
    LLMResponse[] responses;

    foreach (item; arr) {
        string[] objs;
        foreach (o; item["identified_objects"].array)
            objs ~= o.toString();

        responses ~= LLMResponse(item["sample_id"].toString(),
                item["raw_output"].toString(), objs,
                item["decision"].toString(), item["reasoning"].toString());
    }

    responses.sort!((a, b) => a.sampleId < b.sampleId);
    return responses;
}

// ============================================================
// REPORT
// ============================================================

void printReport(GroundTruthSample[] samples, LLMResponse[] responses,
        PerceptionMetrics perc, StrategyMetrics strat, SampleScore[] scores) {
    writeln("\n", strRepeat("=", 64));
    writeln("  LLM STRATEGIC VISION EVALUATION");
    writeln(strRepeat("=", 64));

    double avgPerc = 0.0, avgStrat = 0.0, avgCombined = 0.0;
    if (scores.length > 0) {
        double sumPerc = 0, sumStrat = 0, sumCombined = 0;
        foreach (s; scores) {
            sumPerc += s.perceptionIou;
            sumStrat += s.strategyCorrect ? 1.0 : 0.0;
            sumCombined += s.combinedScore;
        }
        avgPerc = sumPerc / scores.length;
        avgStrat = sumStrat / scores.length;
        avgCombined = sumCombined / scores.length;
    }
    writefln("\nSamples: %d/%d", scores.length, samples.length);
    writefln("Perception (avg IoU):  %.4f", avgPerc);
    writefln("Strategy (accuracy):   %.4f", avgStrat);
    writefln("Combined score:        %.4f", avgCombined);

    writefln("\n--- Perception Metrics ---");
    writefln("  Object Precision:     %.4f", perc.precision);
    writefln("  Object Recall:        %.4f", perc.recall);
    writefln("  Object F1:            %.4f", perc.f1);
    writefln("  Exact Match Rate:     %.4f", perc.exactMatchRate);

    if (perc.objectTypes.length > 0) {
        writefln("\n  Per-Object Breakdown:");
        foreach (i, obj; perc.objectTypes) {
            writefln("    %-25s  P=%.3f  R=%.3f", obj,
                    perc.objectPrecisions[i], perc.objectRecalls[i]);
        }
    }

    writefln("\n--- Strategy Metrics ---");
    writefln("  Decision Accuracy:    %.4f", strat.decisionAccuracy);
    writefln("  Rule Recall:          %.4f", strat.ruleRecall);
    writefln("  Rule Precision:       %.4f", strat.rulePrecision);
    writefln("  Hallucinated Rules:   %d", strat.hallucinatedRules);
    writefln("  Missed Rules:         %d", strat.missedRules);

    int[string] errorCounts;
    foreach (s; scores) {
        errorCounts[s.errorType] = (s.errorType in errorCounts) ? errorCounts[s.errorType] + 1 : 1;
    }

    bool hasErrors = false;
    string[] checkTypes = ["perception_fail", "strategy_fail", "both"];
    foreach (et; checkTypes) {
        if (et in errorCounts && errorCounts[et] > 0)
            hasErrors = true;
    }

    if (hasErrors) {
        writefln("\n--- Error Breakdown ---");
        string[] allTypes = ["perception_fail", "strategy_fail", "both", "none"];
        foreach (et; allTypes) {
            if (et in errorCounts && errorCounts[et] > 0)
                writefln("  %-20s: %d", et, errorCounts[et]);
        }
    }

    SampleScore[] failures;
    foreach (s; scores) {
        if (s.errorType != "none")
            failures ~= s;
    }
    if (!failures.empty) {
        failures.sort!((a, b) => a.combinedScore < b.combinedScore);
        writefln("\n--- Top Failures ---");
        size_t limit = failures.length.min(5);
        foreach (f; failures[0 .. limit]) {
            writefln("\n  [%s] error=%s", f.sampleId, f.errorType);
            writefln("    Objects:  GT=%s", f.gtObjects);
            writefln("             Pred=%s", f.predObjects);
            writefln("    Decision: GT=%s", f.gtDecision);
            writefln("             Pred=%s", f.predDecision);
        }
    }

    writefln("\n%s", strRepeat("=", 64));
}

// ============================================================
// MAIN
// ============================================================

void main(string[] args) {
    string datasetPath = "";
    string responsesPath = "";
    bool showReport = false;

    for (size_t i = 1; i < args.length; i++) {
        if (args[i] == "--dataset" && i + 1 < args.length) {
            datasetPath = args[++i];
        } else if (args[i] == "--responses" && i + 1 < args.length) {
            responsesPath = args[++i];
        } else if (args[i] == "--report") {
            showReport = true;
        }
    }

    if (datasetPath == "" || responsesPath == "") {
        writeln("Usage: evaluate --dataset <dataset.json> --responses <responses.json> [--report]");
        return;
    }

    auto samples = loadDataset(datasetPath);
    auto responses = loadResponses(responsesPath);

    // Align by sample_id
    LLMResponse[string] respMap;
    foreach (r; responses)
        respMap[r.sampleId] = r;

    GroundTruthSample[] alignedSamples;
    LLMResponse[] alignedResponses;
    foreach (s; samples) {
        if (s.sampleId in respMap) {
            alignedSamples ~= s;
            alignedResponses ~= respMap[s.sampleId];
        }
    }

    auto perc = computePerceptionMetrics(alignedSamples, alignedResponses);
    auto strat = computeStrategyMetrics(alignedSamples, alignedResponses);
    auto scores = scoreSamples(alignedSamples, alignedResponses);

    if (showReport)
        printReport(samples, alignedResponses, perc, strat, scores);
    else
        writeln("Evaluation complete. %d/%d samples evaluated. Use --report for details.",
                alignedSamples.length, samples.length);
}
