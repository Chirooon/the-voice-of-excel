"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import {
  Upload,
  Mic,
  Volume2,
  FileSpreadsheet,
  AlertCircle,
  BarChart4,
  Brain,
  Wand2,
  Download,
  Globe,
} from "lucide-react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { read, utils } from "xlsx"
import * as tf from "@tensorflow/tfjs"
import "@tensorflow/tfjs-backend-webgl"
import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Microphone Level Indicator Component
const MicrophoneLevelIndicator = ({ isListening }: { isListening: boolean }) => {
  const [level, setLevel] = useState(0)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  useEffect(() => {
    if (!isListening) {
      setLevel(0)
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      return
    }

    // Get the global analyser node
    const analyser = window.microphoneAnalyser
    if (!analyser) return

    analyserRef.current = analyser
    const dataArray = new Uint8Array(analyser.frequencyBinCount)

    const updateLevel = () => {
      if (!analyserRef.current) return

      analyserRef.current.getByteFrequencyData(dataArray)

      // Calculate average level
      const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length
      const normalizedLevel = Math.min(100, Math.max(0, average * 1.5)) // Scale for better visibility

      setLevel(normalizedLevel)
      animationFrameRef.current = requestAnimationFrame(updateLevel)
    }

    updateLevel()

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
    }
  }, [isListening])

  return (
    <div className="flex items-center gap-2">
      <div className="h-6 w-32 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full ${level > 50 ? "bg-green-500" : "bg-blue-500"} transition-all duration-100`}
          style={{ width: `${level}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground">
        {level < 10 ? "No voice detected" : level < 30 ? "Low volume" : level < 60 ? "Good" : "Excellent"}
      </span>
    </div>
  )
}

// Types
type Status = "idle" | "uploading" | "uploaded" | "listening" | "processing" | "speaking" | "error" | "loading-model"
type Language = "en-US" | "de-DE" | "auto" | null
type ExcelData = {
  sheets: Record<string, any[]>
  activeSheet: string
}
type QueryResult = {
  answer: string
  explanation: string
  data: any[] | null
  followUpQuestions: string[]
  confidence: number
  usedColumns: string[]
  operation: string
}
type SpeechSettings = {
  noiseReduction: boolean
  echoReduction: boolean
  multiSpeaker: boolean
  sensitivity: number
  rate: number
  pitch: number
  volume: number
}
type VisualizationType = "table" | "chart" | null
type NLPModel = {
  model: any
  encoder: any
  tokenizer: any
  loaded: boolean
}

// Declare SpeechRecognition
declare var webkitSpeechRecognition: any

// Add global microphone analyser
declare global {
  interface Window {
    microphoneAnalyser: AnalyserNode | null
  }
}

// Initialize global microphone analyser
if (typeof window !== "undefined") {
  window.microphoneAnalyser = null
}

// Function to extract column names from query
const extractColumns = (query: string, columns: string[]): string[] => {
  const mentionedColumns: string[] = []
  const lowerCaseQuery = query.toLowerCase()

  for (const column of columns) {
    if (lowerCaseQuery.includes(column.toLowerCase())) {
      mentionedColumns.push(column)
    }
  }

  return mentionedColumns
}

// Free NLP processing using TensorFlow.js
const processQueryWithLocalAI = async (
  query: string,
  excelData: ExcelData,
  language: Language,
  nlpModel: NLPModel,
): Promise<QueryResult> => {
  try {
    if (!nlpModel.loaded) {
      throw new Error("NLP model not loaded")
    }

    const currentSheet = excelData.sheets[excelData.activeSheet]
    const columns = currentSheet.length > 0 ? Object.keys(currentSheet[0]) : []

    // Basic intent recognition
    const intent = recognizeIntent(query.toLowerCase())

    // Extract column names from query
    const mentionedColumns = extractColumns(query, columns)

    // Process the query based on intent
    let result: QueryResult

    switch (intent) {
      case "average":
        result = calculateAverage(query, currentSheet, mentionedColumns, language)
        break
      case "sum":
        result = calculateSum(query, currentSheet, mentionedColumns, language)
        break
      case "count":
        result = countValues(query, currentSheet, mentionedColumns, language)
        break
      case "min":
        result = findMinimum(query, currentSheet, mentionedColumns, language)
        break
      case "max":
        result = findMaximum(query, currentSheet, mentionedColumns, language)
        break
      case "correlation":
        result = calculateCorrelation(query, currentSheet, mentionedColumns, language)
        break
      case "unique":
        result = findUniqueValues(query, currentSheet, mentionedColumns, language)
        break
      case "overview":
        result = getDataOverview(currentSheet, columns, language)
        break
      case "search":
        result = searchForValue(query, currentSheet, columns, language)
        break
      default:
        // If no specific intent is recognized, try search as a fallback
        result = searchForValue(query, currentSheet, columns, language)

        // If search doesn't yield good results, provide a general response
        if (result.confidence < 0.5) {
          result = {
            answer:
              language === "de-DE"
                ? "Ich verstehe Ihre Anfrage nicht vollständig. Sie können nach Werten suchen, oder Fragen über Durchschnitt, Summe, Anzahl, Minimum, Maximum oder Korrelation stellen."
                : "I don't fully understand your query. You can search for values, or ask about average, sum, count, minimum, maximum, or correlation.",
            explanation: "",
            data: null,
            followUpQuestions: generateFollowUpQuestions(columns, language),
            confidence: 0.3,
            usedColumns: [],
            operation: "unknown",
          }
        }
    }

    return result
  } catch (error) {
    console.error("Local AI processing error:", error)
    return {
      answer:
        language === "de-DE"
          ? "Es gab einen Fehler bei der Verarbeitung Ihrer Anfrage."
          : "There was an error processing your query.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0,
      usedColumns: [],
      operation: "error",
    }
  }
}

// Enhanced intent recognition
const recognizeIntent = (query: string): string => {
  // Define intent patterns with multiple keywords for each intent
  const intentPatterns = {
    average: [
      "average",
      "mean",
      "avg",
      "durchschnitt",
      "mittelwert",
      "moyenne",
      "media",
      "promedio",
      "プロmedio",
      "平均",
      "what is the average",
      "calculate the average",
      "find the mean",
      "show me the average",
    ],
    sum: [
      "sum",
      "total",
      "add",
      "summe",
      "gesamt",
      "somme",
      "total",
      "suma",
      "合計",
      "总和",
      "what is the sum",
      "calculate the sum",
      "add up",
      "total of",
      "sum of",
    ],
    count: [
      "count",
      "how many",
      "number of",
      "anzahl",
      "wie viele",
      "combien",
      "cuántos",
      "quanti",
      "数える",
      "计数",
      "count the",
      "how many are there",
      "number of items",
      "count all",
    ],
    min: [
      "minimum",
      "min",
      "smallest",
      "lowest",
      "kleinste",
      "minimum",
      "mínimo",
      "minimo",
      "最小",
      "最小值",
      "what is the minimum",
      "find the lowest",
      "what's the smallest",
      "show me the minimum",
    ],
    max: [
      "maximum",
      "max",
      "largest",
      "highest",
      "größte",
      "maximum",
      "máximo",
      "massimo",
      "最大",
      "最大值",
      "what is the maximum",
      "find the highest",
      "what's the largest",
      "show me the maximum",
    ],
    correlation: [
      "correlation",
      "relationship",
      "relates to",
      "korrelation",
      "zusammenhang",
      "corrélation",
      "correlación",
      "correlazione",
      "相関",
      "相关性",
      "how does * relate to",
      "relationship between",
      "correlation between",
      "how * correlates with",
    ],
    unique: [
      "unique",
      "distinct",
      "different",
      "einzigartig",
      "unterschiedlich",
      "unique",
      "único",
      "unico",
      "ユニーク",
      "唯一",
      "unique values",
      "distinct values",
      "list of unique",
      "all different",
      "show me unique",
    ],
    overview: [
      "overview",
      "summary",
      "describe",
      "überblick",
      "zusammenfassung",
      "aperçu",
      "resumen",
      "panoramica",
      "概要",
      "概述",
      "give me an overview",
      "summarize the data",
      "describe the dataset",
      "what's in the data",
    ],
    search: [
      "is",
      "find",
      "search",
      "look for",
      "has",
      "contains",
      "where",
      "ist",
      "finde",
      "suche",
      "enthält",
      "wo",
      "rechercher",
      "buscar",
      "cercare",
      "検索",
      "搜索",
      "find records where",
      "search for rows with",
      "look for entries",
    ],
    filter: [
      "filter",
      "only show",
      "filtern",
      "nur zeigen",
      "filtrer",
      "filtrar",
      "filtrare",
      "フィルター",
      "筛选",
      "filter by",
      "only display",
      "show only",
      "exclude",
    ],
    trend: [
      "trend",
      "pattern",
      "over time",
      "change",
      "trend",
      "muster",
      "im laufe der zeit",
      "änderung",
      "tendance",
      "tendencia",
      "andamento",
      "トレンド",
      "趋势",
      "show me the trend",
      "how has * changed",
      "pattern of",
    ],
  }

  // Convert query to lowercase for case-insensitive matching
  const lowerQuery = query.toLowerCase()

  // Scoring system for intent matching
  let bestIntent = "unknown"
  let highestScore = 0

  for (const [intent, patterns] of Object.entries(intentPatterns)) {
    // Count how many patterns match in the query
    let matchCount = 0
    for (const pattern of patterns) {
      // Use wildcard pattern matching if pattern contains *
      if (pattern.includes("*")) {
        const parts = pattern.split("*").filter((p) => p.length > 0)
        if (parts.length > 0 && parts.every((part) => lowerQuery.includes(part))) {
          matchCount += 1
        }
      }
      // Otherwise use simple inclusion check
      else if (lowerQuery.includes(pattern)) {
        matchCount += 1
      }
    }

    // Calculate score based on number of matches and pattern specificity
    const score = matchCount * (patterns.some((p) => p.includes(" ")) ? 1.5 : 1)

    if (score > highestScore) {
      highestScore = score
      bestIntent = intent
    }
  }

  // If confidence is too low, default to search or unknown
  return highestScore > 0 ? bestIntent : "unknown"
}

// Add a new function to search for values in the data
const searchForValue = (query: string, data: any[], columns: string[], language: Language): QueryResult => {
  // Extract potential search values (numbers, IDs, names, etc.)
  const searchValues = extractSearchValues(query)

  if (searchValues.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? "Ich konnte keinen Suchwert in Ihrer Anfrage erkennen. Bitte geben Sie an, wonach Sie suchen möchten."
          : "I couldn't identify a search value in your query. Please specify what you're looking for.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.5,
      usedColumns: [],
      operation: "search",
    }
  }

  // Identify potential column to search in
  let targetColumn = ""
  const columnKeywords = extractColumnKeywords(query)

  // Try to match column keywords with actual column names
  for (const keyword of columnKeywords) {
    for (const column of columns) {
      if (column.toLowerCase().includes(keyword.toLowerCase())) {
        targetColumn = column
        break
      }
    }
    if (targetColumn) break
  }

  // If no specific column is identified, search across all columns
  const searchResults = []
  const matchingRows = []
  const searchValue = searchValues[0] // Use the first identified search value

  if (targetColumn) {
    // Search in the specific column
    for (const row of data) {
      const cellValue = String(row[targetColumn]).toLowerCase()
      if (cellValue === String(searchValue).toLowerCase()) {
        matchingRows.push(row)
      }
    }
  } else {
    // Search across all columns
    for (const row of data) {
      for (const column of columns) {
        const cellValue = String(row[column]).toLowerCase()
        if (cellValue === String(searchValue).toLowerCase()) {
          matchingRows.push(row)
          if (!searchResults.includes(column)) {
            searchResults.push(column)
          }
        }
      }
    }
  }

  // Prepare the response
  if (matchingRows.length > 0) {
    const foundColumns = targetColumn ? [targetColumn] : searchResults

    return {
      answer:
        language === "de-DE"
          ? `Ja, ${searchValue} wurde in den Daten gefunden. Es gibt ${matchingRows.length} Übereinstimmung(en).`
          : `Yes, ${searchValue} was found in the data. There are ${matchingRows.length} match(es).`,
      explanation:
        language === "de-DE"
          ? `Ich habe nach "${searchValue}" in ${targetColumn ? `der Spalte "${targetColumn}"` : "allen Spalten"} gesucht und ${matchingRows.length} Übereinstimmung(en) gefunden.`
          : `I searched for "${searchValue}" in ${targetColumn ? `the "${targetColumn}" column` : "all columns"} and found ${matchingRows.length} match(es).`,
      data: matchingRows.slice(0, 10),
      followUpQuestions: generateSearchFollowUpQuestions(searchValue, matchingRows, columns, language),
      confidence: 0.9,
      usedColumns: foundColumns,
      operation: "search",
    }
  } else {
    return {
      answer:
        language === "de-DE"
          ? `Nein, ${searchValue} wurde nicht in den Daten gefunden.`
          : `No, ${searchValue} was not found in the data.`,
      explanation:
        language === "de-DE"
          ? `Ich habe nach "${searchValue}" in ${targetColumn ? `der Spalte "${targetColumn}"` : "allen Spalten"} gesucht, aber keine Übereinstimmungen gefunden.`
          : `I searched for "${searchValue}" in ${targetColumn ? `the "${targetColumn}" column` : "all columns"} but found no matches.`,
      data: null,
      followUpQuestions: [
        language === "de-DE" ? "Gib mir einen Überblick über die Daten." : "Give me an overview of the data.",
        language === "de-DE" ? "Welche Werte gibt es in dieser Spalte?" : "What values exist in this column?",
        language === "de-DE" ? "Wie viele Einträge gibt es insgesamt?" : "How many entries are there in total?",
      ],
      confidence: 0.9,
      usedColumns: targetColumn ? [targetColumn] : [],
      operation: "search",
    }
  }
}

// Extract search values from the query
const extractSearchValues = (query: string): string[] => {
  const values = []

  // Extract numbers (including those that might be IDs)
  const numberMatches = query.match(/\d+/g)
  if (numberMatches) {
    values.push(...numberMatches)
  }

  // Extract potential quoted values
  const quotedMatches = query.match(/"([^"]*)"|'([^']*)'/g)
  if (quotedMatches) {
    values.push(...quotedMatches.map((match) => match.slice(1, -1)))
  }

  // Extract potential ID values (common patterns like ID123, id-456, etc.)
  const idMatches = query.match(/[a-zA-Z]+-?\d+/g)
  if (idMatches) {
    values.push(...idMatches)
  }

  return values
}

// Extract column keywords from the query
const extractColumnKeywords = (query: string): string[] => {
  const keywords = []

  // Common column identifiers
  const columnIdentifiers = [
    "id",
    "name",
    "title",
    "date",
    "time",
    "price",
    "cost",
    "value",
    "number",
    "email",
    "phone",
    "address",
    "city",
    "state",
    "country",
    "category",
    "type",
    "status",
    "description",
    "quantity",
    "amount",
  ]

  // Check for column identifiers in the query
  for (const identifier of columnIdentifiers) {
    if (query.toLowerCase().includes(identifier.toLowerCase())) {
      keywords.push(identifier)
    }
  }

  return keywords
}

// Generate follow-up questions for search results
const generateSearchFollowUpQuestions = (
  searchValue: string,
  matchingRows: any[],
  columns: string[],
  language: Language,
): string[] => {
  const questions = []

  if (matchingRows.length > 0) {
    // Get a random column from the first matching row
    const availableColumns = Object.keys(matchingRows[0])
    const randomColumn = availableColumns[Math.floor(Math.random() * availableColumns.length)]

    if (language === "de-DE") {
      questions.push(`Wie viele Einträge haben den Wert ${searchValue}?`)
      questions.push(`Was ist der Durchschnitt von ${randomColumn}?`)
      questions.push(`Gibt es andere Einträge mit ähnlichen Werten?`)
    } else {
      questions.push(`How many entries have the value ${searchValue}?`)
      questions.push(`What is the average of ${randomColumn}?`)
      questions.push(`Are there other entries with similar values?`)
    }
  } else {
    if (language === "de-DE") {
      questions.push("Welche Werte gibt es in dieser Spalte?")
      questions.push("Gib mir einen Überblick über die Daten.")
      questions.push("Wie viele Einträge gibt es insgesamt?")
    } else {
      questions.push("What values exist in this column?")
      questions.push("Give me an overview of the data.")
      questions.push("How many entries are there in total?")
    }
  }

  return questions
}

// Calculate average
const calculateAverage = (query: string, data: any[], mentionedColumns: string[], language: Language): QueryResult => {
  if (mentionedColumns.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? "Bitte geben Sie eine Spalte an, für die Sie den Durchschnitt berechnen möchten."
          : "Please specify a column for which you want to calculate the average.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.5,
      usedColumns: [],
      operation: "average",
    }
  }

  const column = mentionedColumns[0]
  const values = data.map((row) => Number.parseFloat(row[column])).filter((val) => !isNaN(val))

  if (values.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? `Ich konnte keine numerischen Werte in der Spalte ${column} finden.`
          : `I couldn't find any numeric values in the ${column} column.`,
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.7,
      usedColumns: [column],
      operation: "average",
    }
  }

  const average = values.reduce((sum, val) => sum + val, 0) / values.length

  return {
    answer:
      language === "de-DE"
        ? `Der Durchschnitt von ${column} ist ${average.toFixed(2)}.`
        : `The average of ${column} is ${average.toFixed(2)}.`,
    explanation:
      language === "de-DE"
        ? `Ich habe den Durchschnitt berechnet, indem ich alle Werte in der Spalte ${column} addiert und durch die Anzahl der Werte (${values.length}) geteilt habe.`
        : `I calculated the average by adding all values in the ${column} column and dividing by the number of values (${values.length}).`,
    data: data.map((row) => ({ [column]: row[column] })).slice(0, 10),
    followUpQuestions: [
      language === "de-DE" ? `Was ist die Summe von ${column}?` : `What is the sum of ${column}?`,
      language === "de-DE" ? `Was ist der Maximalwert von ${column}?` : `What is the maximum value of ${column}?`,
      language === "de-DE" ? `Wie viele Einträge gibt es in ${column}?` : `How many entries are there in ${column}?`,
    ],
    confidence: 0.9,
    usedColumns: [column],
    operation: "average",
  }
}

// Calculate sum
const calculateSum = (query: string, data: any[], mentionedColumns: string[], language: Language): QueryResult => {
  if (mentionedColumns.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? "Bitte geben Sie eine Spalte an, für die Sie die Summe berechnen möchten."
          : "Please specify a column for which you want to calculate the sum.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.5,
      usedColumns: [],
      operation: "sum",
    }
  }

  const column = mentionedColumns[0]
  const values = data.map((row) => Number.parseFloat(row[column])).filter((val) => !isNaN(val))

  if (values.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? `Ich konnte keine numerischen Werte in der Spalte ${column} finden.`
          : `I couldn't find any numeric values in the ${column} column.`,
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.7,
      usedColumns: [column],
      operation: "sum",
    }
  }

  const sum = values.reduce((total, val) => total + val, 0)

  return {
    answer:
      language === "de-DE"
        ? `Die Summe von ${column} ist ${sum.toFixed(2)}.`
        : `The sum of ${column} is ${sum.toFixed(2)}.`,
    explanation:
      language === "de-DE"
        ? `Ich habe die Summe berechnet, indem ich alle Werte in der Spalte ${column} addiert habe.`
        : `I calculated the sum by adding all values in the ${column} column.`,
    data: data.map((row) => ({ [column]: row[column] })).slice(0, 10),
    followUpQuestions: [
      language === "de-DE" ? `Was ist der Durchschnitt von ${column}?` : `What is the average of ${column}?`,
      language === "de-DE" ? `Was ist der Minimalwert von ${column}?` : `What is the minimum value of ${column}?`,
      language === "de-DE" ? `Wie viele Einträge gibt es in ${column}?` : `How many entries are there in ${column}?`,
    ],
    confidence: 0.9,
    usedColumns: [column],
    operation: "sum",
  }
}

// Count values
const countValues = (query: string, data: any[], mentionedColumns: string[], language: Language): QueryResult => {
  if (mentionedColumns.length === 0) {
    // Count all rows
    return {
      answer:
        language === "de-DE"
          ? `Es gibt insgesamt ${data.length} Einträge in dieser Tabelle.`
          : `There are a total of ${data.length} entries in this sheet.`,
      explanation:
        language === "de-DE"
          ? "Ich habe die Gesamtzahl der Zeilen in der Tabelle gezählt."
          : "I counted the total number of rows in the sheet.",
      data: null,
      followUpQuestions: [],
      confidence: 0.9,
      usedColumns: [],
      operation: "count",
    }
  }

  const column = mentionedColumns[0]
  const nonEmptyCount = data.filter(
    (row) => row[column] !== undefined && row[column] !== null && row[column] !== "",
  ).length

  return {
    answer:
      language === "de-DE"
        ? `Es gibt ${nonEmptyCount} nicht-leere Einträge in der Spalte ${column}.`
        : `There are ${nonEmptyCount} non-empty entries in the ${column} column.`,
    explanation:
      language === "de-DE"
        ? `Ich habe die Anzahl der nicht-leeren Werte in der Spalte ${column} gezählt.`
        : `I counted the number of non-empty values in the ${column} column.`,
    data: data.map((row) => ({ [column]: row[column] })).slice(0, 10),
    followUpQuestions: [
      language === "de-DE" ? `Was ist der Durchschnitt von ${column}?` : `What is the average of ${column}?`,
      language === "de-DE"
        ? `Was sind die einzigartigen Werte in ${column}?`
        : `What are the unique values in ${column}?`,
      language === "de-DE" ? `Wie viele Einträge gibt es insgesamt?` : `How many entries are there in total?`,
    ],
    confidence: 0.9,
    usedColumns: [column],
    operation: "count",
  }
}

// Find minimum value
const findMinimum = (query: string, data: any[], mentionedColumns: string[], language: Language): QueryResult => {
  if (mentionedColumns.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? "Bitte geben Sie eine Spalte an, für die Sie den Minimalwert finden möchten."
          : "Please specify a column for which you want to find the minimum value.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.5,
      usedColumns: [],
      operation: "min",
    }
  }

  const column = mentionedColumns[0]
  const values = data.map((row) => Number.parseFloat(row[column])).filter((val) => !isNaN(val))

  if (values.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? `Ich konnte keine numerischen Werte in der Spalte ${column} finden.`
          : `I couldn't find any numeric values in the ${column} column.`,
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.7,
      usedColumns: [column],
      operation: "min",
    }
  }

  const min = Math.min(...values)

  return {
    answer:
      language === "de-DE" ? `Der Minimalwert von ${column} ist ${min}.` : `The minimum value of ${column} is ${min}.`,
    explanation:
      language === "de-DE"
        ? `Ich habe den kleinsten Wert in der Spalte ${column} gefunden.`
        : `I found the smallest value in the ${column} column.`,
    data: data.filter((row) => Number.parseFloat(row[column]) === min).slice(0, 10),
    followUpQuestions: [
      language === "de-DE" ? `Was ist der Maximalwert von ${column}?` : `What is the maximum value of ${column}?`,
      language === "de-DE" ? `Was ist der Durchschnitt von ${column}?` : `What is the average of ${column}?`,
      language === "de-DE" ? `Wie viele Einträge haben den Minimalwert?` : `How many entries have the minimum value?`,
    ],
    confidence: 0.9,
    usedColumns: [column],
    operation: "min",
  }
}

// Find maximum value
const findMaximum = (query: string, data: any[], mentionedColumns: string[], language: Language): QueryResult => {
  if (mentionedColumns.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? "Bitte geben Sie eine Spalte an, für die Sie den Maximalwert finden möchten."
          : "Please specify a column for which you want to find the maximum value.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.5,
      usedColumns: [],
      operation: "max",
    }
  }

  const column = mentionedColumns[0]
  const values = data.map((row) => Number.parseFloat(row[column])).filter((val) => !isNaN(val))

  if (values.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? `Ich konnte keine numerischen Werte in der Spalte ${column} finden.`
          : `I couldn't find any numeric values in the ${column} column.`,
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.7,
      usedColumns: [column],
      operation: "max",
    }
  }

  const max = Math.max(...values)

  return {
    answer:
      language === "de-DE" ? `Der Maximalwert von ${column} ist ${max}.` : `The maximum value of ${column} is ${max}.`,
    explanation:
      language === "de-DE"
        ? `Ich habe den größten Wert in der Spalte ${column} gefunden.`
        : `I found the largest value in the ${column} column.`,
    data: data.filter((row) => Number.parseFloat(row[column]) === max).slice(0, 10),
    followUpQuestions: [
      language === "de-DE" ? `Was ist der Minimalwert von ${column}?` : `What is the minimum value of ${column}?`,
      language === "de-DE" ? `Was ist der Durchschnitt von ${column}?` : `What is the average of ${column}?`,
      language === "de-DE" ? `Wie viele Einträge haben den Maximalwert?` : `How many entries have the maximum value?`,
    ],
    confidence: 0.9,
    usedColumns: [column],
    operation: "max",
  }
}

// Calculate correlation
const calculateCorrelation = (
  query: string,
  data: any[],
  mentionedColumns: string[],
  language: Language,
): QueryResult => {
  if (mentionedColumns.length < 2) {
    return {
      answer:
        language === "de-DE"
          ? "Bitte geben Sie zwei Spalten an, zwischen denen Sie die Korrelation berechnen möchten."
          : "Please specify two columns between which you want to calculate correlation.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.5,
      usedColumns: mentionedColumns,
      operation: "correlation",
    }
  }

  const column1 = mentionedColumns[0]
  const column2 = mentionedColumns[1]

  // Get numeric values from both columns
  const values1 = data.map((row) => Number.parseFloat(row[column1])).filter((val) => !isNaN(val))
  const values2 = data.map((row) => Number.parseFloat(row[column2])).filter((val) => !isNaN(val))

  // Need matching pairs of values
  const minLength = Math.min(values1.length, values2.length)

  if (minLength < 5) {
    return {
      answer:
        language === "de-DE"
          ? `Es gibt nicht genügend numerische Daten in ${column1} und ${column2}, um eine Korrelation zu berechnen.`
          : `There isn't enough numeric data in ${column1} and ${column2} to calculate correlation.`,
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.7,
      usedColumns: [column1, column2],
      operation: "correlation",
    }
  }

  // Calculate correlation coefficient (Pearson)
  const mean1 = values1.reduce((sum, val) => sum + val, 0) / values1.length
  const mean2 = values2.reduce((sum, val) => sum + val, 0) / values2.length

  let numerator = 0
  let denominator1 = 0
  let denominator2 = 0

  for (let i = 0; i < minLength; i++) {
    const diff1 = values1[i] - mean1
    const diff2 = values2[i] - mean2

    numerator += diff1 * diff2
    denominator1 += diff1 * diff1
    denominator2 += diff2 * diff2
  }

  const correlation = numerator / (Math.sqrt(denominator1) * Math.sqrt(denominator2))

  // Interpret correlation
  let interpretation = ""
  if (Math.abs(correlation) < 0.3) {
    interpretation = language === "de-DE" ? "schwache" : "weak"
  } else if (Math.abs(correlation) < 0.7) {
    interpretation = language === "de-DE" ? "moderate" : "moderate"
  } else {
    interpretation = language === "de-DE" ? "starke" : "strong"
  }

  const direction =
    correlation > 0 ? (language === "de-DE" ? "positive" : "positive") : language === "de-DE" ? "negative" : "negative"

  return {
    answer:
      language === "de-DE"
        ? `Die Korrelation zwischen ${column1} und ${column2} ist ${correlation.toFixed(2)}, was auf eine ${interpretation} ${direction} Beziehung hinweist.`
        : `The correlation between ${column1} and ${column2} is ${correlation.toFixed(2)}, indicating a ${interpretation} ${direction} relationship.`,
    explanation:
      language === "de-DE"
        ? `Ich habe den Pearson-Korrelationskoeffizienten zwischen den Spalten ${column1} und ${column2} berechnet. Ein Wert nahe 1 bedeutet eine starke positive Korrelation, ein Wert nahe -1 bedeutet eine starke negative Korrelation, und ein Wert nahe 0 bedeutet keine Korrelation.`
        : `I calculated the Pearson correlation coefficient between the ${column1} and ${column2} columns. A value close to 1 indicates a strong positive correlation, a value close to -1 indicates a strong negative correlation, and a value close to 0 indicates no correlation.`,
    data: data.map((row) => ({ [column1]: row[column1], [column2]: row[column2] })).slice(0, 10),
    followUpQuestions: [
      language === "de-DE" ? `Was ist der Durchschnitt von ${column1}?` : `What is the average of ${column1}?`,
      language === "de-DE" ? `Was ist der Durchschnitt von ${column2}?` : `What is the average of ${column2}?`,
      language === "de-DE"
        ? `Gibt es andere Spalten, die mit ${column1} korrelieren?`
        : `Are there other columns that correlate with ${column1}?`,
    ],
    confidence: 0.9,
    usedColumns: [column1, column2],
    operation: "correlation",
  }
}

// Find unique values
const findUniqueValues = (query: string, data: any[], mentionedColumns: string[], language: Language): QueryResult => {
  if (mentionedColumns.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? "Bitte geben Sie eine Spalte an, für die Sie einzigartige Werte finden möchten."
          : "Please specify a column for which you want to find unique values.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.5,
      usedColumns: [],
      operation: "unique",
    }
  }

  const column = mentionedColumns[0]
  const allValues = data.map((row) => row[column])
  const uniqueValues = [...new Set(allValues.filter((val) => val !== undefined && val !== null && val !== ""))]

  if (uniqueValues.length === 0) {
    return {
      answer:
        language === "de-DE"
          ? `Die Spalte ${column} enthält keine Werte oder ist leer.`
          : `The ${column} column contains no values or is empty.`,
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0.7,
      usedColumns: [column],
      operation: "unique",
    }
  }

  // Limit the number of values to display
  const displayLimit = 10
  const displayValues = uniqueValues.slice(0, displayLimit).join(", ")
  const remainingCount = uniqueValues.length - displayLimit

  let valueList = displayValues
  if (remainingCount > 0) {
    valueList += language === "de-DE" ? ` und ${remainingCount} weitere` : ` and ${remainingCount} more`
  }

  return {
    answer:
      language === "de-DE"
        ? `Die Spalte ${column} enthält ${uniqueValues.length} einzigartige Werte: ${valueList}.`
        : `The ${column} column contains ${uniqueValues.length} unique values: ${valueList}.`,
    explanation:
      language === "de-DE"
        ? `Ich habe alle einzigartigen Werte in der Spalte ${column} gefunden und gezählt.`
        : `I found and counted all unique values in the ${column} column.`,
    data: uniqueValues.slice(0, 20).map((value) => ({ [column]: value })),
    followUpQuestions: [
      language === "de-DE" ? `Wie viele Einträge gibt es in ${column}?` : `How many entries are there in ${column}?`,
      language === "de-DE" ? `Was ist der häufigste Wert in ${column}?` : `What is the most common value in ${column}?`,
      language === "de-DE"
        ? `Wie ist die Verteilung der Werte in ${column}?`
        : `What is the distribution of values in ${column}?`,
    ],
    confidence: 0.9,
    usedColumns: [column],
    operation: "unique",
  }
}

// Get data overview
const getDataOverview = (data: any[], columns: string[], language: Language): QueryResult => {
  const numRows = data.length
  const numCols = columns.length

  // Count numeric columns
  let numericColumns = 0
  for (const col of columns) {
    const values = data.slice(0, 10).map((row) => row[col])
    const allNumbers = values.every((val) => val !== undefined && val !== null && val !== "" && !isNaN(Number(val)))
    if (allNumbers) numericColumns++
  }

  return {
    answer:
      language === "de-DE"
        ? `Tabellenübersicht: ${numRows} Zeilen, ${numCols} Spalten. Davon sind etwa ${numericColumns} Spalten numerisch. Die Spalten sind: ${columns.join(", ")}.`
        : `Sheet overview: ${numRows} rows, ${numCols} columns. Approximately ${numericColumns} columns contain numeric data. The columns are: ${columns.join(", ")}.`,
    explanation:
      language === "de-DE"
        ? "Ich habe die grundlegenden Eigenschaften der Tabelle analysiert, einschließlich der Anzahl der Zeilen, Spalten und der Art der Daten."
        : "I analyzed the basic properties of the sheet, including the number of rows, columns, and the type of data.",
    data: null,
    followUpQuestions: [
      language === "de-DE" ? "Welche Spalten enthalten numerische Daten?" : "Which columns contain numeric data?",
      language === "de-DE"
        ? "Was ist die Spalte mit den meisten einzigartigen Werten?"
        : "What is the column with the most unique values?",
      language === "de-DE"
        ? "Gibt es Korrelationen zwischen den numerischen Spalten?"
        : "Are there correlations between the numeric columns?",
    ],
    confidence: 0.9,
    usedColumns: columns,
    operation: "overview",
  }
}

// Generate follow-up questions
const generateFollowUpQuestions = (columns: string[], language: Language): string[] => {
  const questions: string[] = []

  // Select a few random columns for follow-up questions
  const randomColumns = columns.sort(() => 0.5 - Math.random()).slice(0, Math.min(3, columns.length))

  for (const column of randomColumns) {
    if (language === "de-DE") {
      questions.push(`Was ist der Durchschnitt von ${column}?`)
      questions.push(`Wie viele einzigartige Werte gibt es in ${column}?`)
    } else {
      questions.push(`What is the average of ${column}?`)
      questions.push(`How many unique values are there in ${column}?`)
    }
  }

  // Add general questions
  if (language === "de-DE") {
    questions.push("Gib mir einen Überblick über die Daten.")
    questions.push("Wie viele Einträge gibt es insgesamt?")
  } else {
    questions.push("Give me an overview of the data.")
    questions.push("How many entries are there in total?")
  }

  // Return a subset of questions
  return questions.slice(0, 5)
}

// Load TensorFlow.js model for NLP
const loadNLPModel = async (): Promise<NLPModel> => {
  try {
    // In a real implementation, you would load a pre-trained model
    // For this demo, we'll create a dummy model
    console.log("Loading TensorFlow.js model...")

    // Simulate model loading time
    await new Promise((resolve) => setTimeout(resolve, 1500))

    return {
      model: {},
      encoder: {},
      tokenizer: {},
      loaded: true,
    }
  } catch (error) {
    console.error("Error loading NLP model:", error)
    throw error
  }
}

// Mock AI model for query understanding (in a real app, this would be a call to an actual AI service)
const processQueryWithAI = async (query: string, excelData: ExcelData, language: Language): Promise<QueryResult> => {
  try {
    // In a real implementation, this would be a call to an AI service
    // For this demo, we'll use the AI SDK to simulate the AI processing
    const currentSheet = excelData.sheets[excelData.activeSheet]
    const columns = currentSheet.length > 0 ? Object.keys(currentSheet[0]) : []

    // Create a prompt for the AI
    const prompt = `
You are an expert data analyst assistant. Analyze this Excel data and answer the user's query.

EXCEL DATA SUMMARY:
- Sheet name: ${excelData.activeSheet}
- Number of rows: ${currentSheet.length}
- Columns: ${columns.join(", ")}

Here's a sample of the data (first 3 rows):
${JSON.stringify(currentSheet.slice(0, 3), null, 2)}

USER QUERY: "${query}"
USER LANGUAGE: ${language === "de-DE" ? "German" : "English"}

Provide a response in the following JSON format:
{
  "answer": "The direct answer to the user's query",
  "explanation": "A brief explanation of how you arrived at this answer",
  "data": [Array of relevant data points used for the analysis, or null if not applicable],
  "followUpQuestions": ["3-5 relevant follow-up questions the user might want to ask"],
  "confidence": A number between 0 and 1 indicating your confidence in the answer,
  "usedColumns": ["List of column names used in the analysis"],
  "operation": "The type of operation performed (e.g., average, sum, count, filter, etc.)"
}

Ensure your answer is accurate, concise, and directly addresses the user's query.
`

    // Use the AI SDK to process the query
    // In a production app, you would use a more sophisticated model or service
    const { text } = await generateText({
      model: openai("gpt-3.5-turbo"),
      prompt: prompt,
      temperature: 0.2,
      maxTokens: 1000,
    })

    // Parse the response
    try {
      const parsedResponse = JSON.parse(text)
      return parsedResponse as QueryResult
    } catch (error) {
      console.error("Failed to parse AI response:", error)
      // Fallback response if parsing fails
      return {
        answer:
          language === "de-DE"
            ? "Ich konnte Ihre Anfrage nicht vollständig verstehen. Bitte versuchen Sie es erneut."
            : "I couldn't fully understand your query. Please try again.",
        explanation: "",
        data: null,
        followUpQuestions: [],
        confidence: 0.3,
        usedColumns: [],
        operation: "unknown",
      }
    }
  } catch (error) {
    console.error("AI processing error:", error)
    return {
      answer:
        language === "de-DE"
          ? "Es gab einen Fehler bei der Verarbeitung Ihrer Anfrage."
          : "There was an error processing your query.",
      explanation: "",
      data: null,
      followUpQuestions: [],
      confidence: 0,
      usedColumns: [],
      operation: "error",
    }
  }
}

// Enhanced Excel file reading
const readExcelFile = (file: File, setProgressCallback: (progress: number) => void): Promise<ExcelData> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = async (e) => {
      try {
        const data = e.target?.result

        // Enhanced options for XLSX parsing
        const options = {
          type: "binary",
          cellDates: true, // Properly handle dates
          cellNF: true, // Keep number formats
          cellText: false, // Don't store rich text
          cellStyles: false, // Ignore cell styles for performance
          dateNF: "yyyy-mm-dd", // Date format
          WTF: false, // Don't show formula errors
        }

        const workbook = read(data, options)

        // Progressive processing for large files
        const sheets: Record<string, any[]> = {}

        // Process sheet by sheet
        for (const sheetName of workbook.SheetNames) {
          try {
            const worksheet = workbook.Sheets[sheetName]

            // Convert to JSON with headers
            const jsonData = utils.sheet_to_json(worksheet, {
              header: 1, // Use first row as headers
              defval: null, // Default value for empty cells
              blankrows: false, // Skip blank rows
              raw: false, // Convert values by default
            })

            // Extract headers from first row
            const headers = jsonData[0] as string[]
            const rows = jsonData.slice(1)

            // Convert array data to object structure with proper headers
            const processedData = rows.map((row) => {
              const obj: Record<string, any> = {}
              headers.forEach((header, index) => {
                if (header) {
                  // Skip empty headers
                  obj[header] = row[index] !== undefined ? row[index] : null
                }
              })
              return obj
            })

            // Filter out empty rows
            sheets[sheetName] = processedData.filter((row) =>
              Object.values(row).some((val) => val !== null && val !== ""),
            )
          } catch (sheetError) {
            console.warn(`Error processing sheet ${sheetName}:`, sheetError)
            sheets[sheetName] = [] // Provide empty array for failed sheets
          }
        }

        resolve({
          sheets,
          activeSheet: workbook.SheetNames[0],
        })
      } catch (err) {
        console.error("Excel parsing error:", err)
        reject(new Error(`Failed to parse Excel file: ${err.message}`))
      }
    }

    reader.onerror = (err) => {
      console.error("File reading error:", err)
      reject(new Error("Failed to read file"))
    }

    reader.onprogress = (event) => {
      if (event.lengthComputable) {
        const progress = Math.round((event.loaded / event.total) * 100)
        setProgressCallback(progress)
      }
    }

    // Read file as binary string
    reader.readAsBinaryString(file)
  })
}

// Change the improveAIModelWithContext function to accept nlpModel as a parameter
const improveAIModelWithContext = async (excelData: ExcelData, nlpModel: NLPModel): Promise<void> => {
  if (!nlpModel.loaded || !excelData) return

  console.log("Enhancing AI model with data context...")

  try {
    const currentSheet = excelData.sheets[excelData.activeSheet]
    if (!currentSheet.length) return

    // Extract column statistics for context awareness
    const columnStats: Record<string, any> = {}
    const columns = Object.keys(currentSheet[0])

    for (const column of columns) {
      // Get column values
      const values = currentSheet
        .map((row) => row[column])
        .filter((val) => val !== undefined && val !== null && val !== "")

      // Skip empty columns
      if (!values.length) continue

      // Determine column type
      const numericValues = values.map((v) => Number.parseFloat(String(v))).filter((v) => !isNaN(v))
      const isNumeric = numericValues.length > values.length * 0.5 // More than 50% are numbers

      // Calculate basic statistics for numeric columns
      if (isNumeric) {
        columnStats[column] = {
          type: "numeric",
          count: numericValues.length,
          min: Math.min(...numericValues),
          max: Math.max(...numericValues),
          avg: numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length,
          uniqueCount: new Set(numericValues).size,
        }
      } else {
        // Text analysis for categorical columns
        const uniqueValues = [...new Set(values)]
        columnStats[column] = {
          type: "categorical",
          count: values.length,
          uniqueCount: uniqueValues.length,
          mostCommon: uniqueValues.length <= 20 ? uniqueValues : [], // Only store if reasonable size
        }
      }
    }

    // In a real implementation, this data would be used to train the model
    console.log("Column statistics for AI context:", columnStats)

    // Set model context (simulated)
    ;(nlpModel as any).context = {
      columns,
      statistics: columnStats,
      rowCount: currentSheet.length,
      updated: new Date().toISOString(),
    }

    console.log("AI model enhanced with data context")
  } catch (error) {
    console.error("Error enhancing AI model with context:", error)
  }
}

export default function ExcelVoiceAnalyzer() {
  // State
  const [status, setStatus] = useState<Status>("idle")
  const [language, setLanguage] = useState<Language>("auto")
  const [file, setFile] = useState<File | null>(null)
  const [excelData, setExcelData] = useState<ExcelData | null>(null)
  const [transcript, setTranscript] = useState("")
  const [response, setResponse] = useState<QueryResult | null>(null)
  const [error, setError] = useState("")
  const [uploadProgress, setUploadProgress] = useState(0)
  const [queryHistory, setQueryHistory] = useState<{ query: string; response: QueryResult }[]>([])
  const [activeTab, setActiveTab] = useState("response")
  const [visualization, setVisualization] = useState<VisualizationType>("table")
  const [speechSettings, setSpeechSettings] = useState<SpeechSettings>({
    noiseReduction: true,
    echoReduction: true,
    multiSpeaker: false,
    sensitivity: 0.5,
    rate: 1,
    pitch: 1,
    volume: 1,
  })
  const [isListening, setIsListening] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [detectedLanguage, setDetectedLanguage] = useState<Language>(null)
  const [nlpModel, setNlpModel] = useState<NLPModel>({ model: null, encoder: null, tokenizer: null, loaded: false })
  const [modelLoading, setModelLoading] = useState(false)

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropZoneRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<SpeechRecognition | null>(null)
  const synthRef = useRef<SpeechSynthesis | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const microphoneStreamRef = useRef<MediaStream | null>(null)

  // Add this state for manual input
  const [manualInput, setManualInput] = useState("")

  // Initialize TensorFlow.js
  useEffect(() => {
    const initTensorFlow = async () => {
      try {
        // Initialize TensorFlow.js
        await tf.ready()
        console.log("TensorFlow.js initialized successfully")

        // Load NLP model
        setModelLoading(true)
        const model = await loadNLPModel()
        setNlpModel(model)
        setModelLoading(false)
      } catch (error) {
        console.error("Error initializing TensorFlow.js:", error)
        setError("Failed to initialize AI model. Please try again.")
        setModelLoading(false)
      }
    }

    initTensorFlow()
  }, [])

  // Initialize speech recognition and synthesis
  useEffect(() => {
    if (typeof window !== "undefined") {
      // Speech Recognition setup
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      if (SpeechRecognition) {
        recognitionRef.current = new SpeechRecognition()
        recognitionRef.current.continuous = true
        recognitionRef.current.interimResults = true
      } else {
        setError("Speech recognition is not supported in your browser.")
      }

      // Speech Synthesis setup
      synthRef.current = window.speechSynthesis
      if (!synthRef.current) {
        setError("Speech synthesis is not supported in your browser.")
      }

      // Audio Context for advanced audio processing
      try {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
        analyserRef.current = audioContextRef.current.createAnalyser()
        analyserRef.current.fftSize = 2048
      } catch (err) {
        console.error("Web Audio API is not supported in this browser", err)
      }
    }

    // Drag and drop setup
    const dropZone = dropZoneRef.current
    if (dropZone) {
      const preventDefault = (e: Event) => {
        e.preventDefault()
        e.stopPropagation()
      }

      const handleDragEnter = (e: DragEvent) => {
        preventDefault(e)
        dropZone.classList.add("border-primary")
      }

      const handleDragLeave = (e: DragEvent) => {
        preventDefault(e)
        dropZone.classList.remove("border-primary")
      }

      const handleDragOver = (e: DragEvent) => {
        preventDefault(e)
        dropZone.classList.add("border-primary")
      }

      const handleDrop = (e: DragEvent) => {
        preventDefault(e)
        dropZone.classList.remove("border-primary")

        if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
          handleFileUpload(e.dataTransfer.files[0])
        }
      }

      dropZone.addEventListener("dragenter", handleDragEnter)
      dropZone.addEventListener("dragleave", handleDragLeave)
      dropZone.addEventListener("dragover", handleDragOver)
      dropZone.addEventListener("drop", handleDrop)

      return () => {
        dropZone.removeEventListener("dragenter", handleDragEnter)
        dropZone.removeEventListener("dragleave", handleDragEnter)
        dropZone.removeEventListener("dragover", handleDragOver)
        dropZone.removeEventListener("drop", handleDrop)
      }
    }
  }, [])

  // Start listening after file upload
  useEffect(() => {
    if (excelData && status === "uploaded" && nlpModel.loaded) {
      startListening()
    }
  }, [excelData, status, nlpModel.loaded])

  // Clean up audio resources on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }

      if (synthRef.current) {
        synthRef.current.cancel()
      }

      if (microphoneStreamRef.current) {
        microphoneStreamRef.current.getTracks().forEach((track) => track.stop())
      }

      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [])

  // Move handleFileUpload inside the component
  const handleFileUpload = async (file: File) => {
    if (!file.name.endsWith(".xlsx")) {
      setError("Please upload an Excel (.xlsx) file only.")
      return
    }

    setStatus("uploading")
    setFile(file)
    setError("")

    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          return 100
        }
        return prev + 10
      })
    }, 100)

    try {
      const data = await readExcelFile(file, setUploadProgress)
      setExcelData(data)
      setStatus("uploaded")
      clearInterval(interval)
      setUploadProgress(100)

      // Add this new call to improve AI with context
      if (nlpModel.loaded) {
        improveAIModelWithContext(data, nlpModel)
      }

      // Check if NLP model is loaded
      if (!nlpModel.loaded) {
        setStatus("loading-model")
        setModelLoading(true)
        const model = await loadNLPModel()
        setNlpModel(model)
        setModelLoading(false)
        setStatus("uploaded")
      }
    } catch (err) {
      setError("Failed to parse Excel file. Please try again.")
      setStatus("error")
      clearInterval(interval)
    }
  }

  // Detect language from text
  const detectLanguage = (text: string) => {
    // Language detection based on common German words
    const germanWords = [
      "der",
      "die",
      "das",
      "und",
      "ist",
      "in",
      "zu",
      "den",
      "mit",
      "auf",
      "für",
      "nicht",
      "auch",
      "sich",
      "von",
      "eine",
      "aber",
    ]
    const words = text.toLowerCase().split(" ")
    const germanWordCount = words.filter((word) => germanWords.includes(word)).length

    // If more than 15% of words are German, assume German
    const isGerman = germanWordCount / words.length > 0.15

    const detectedLang = isGerman ? "de-DE" : "en-US"
    setDetectedLanguage(detectedLang)

    // Update recognition language
    if (recognitionRef.current) {
      recognitionRef.current.lang = detectedLang
    }

    return detectedLang
  }

  // Set up advanced audio processing
  const setupAudioProcessing = async () => {
    if (!audioContextRef.current) return

    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: speechSettings.echoReduction,
          noiseSuppression: speechSettings.noiseReduction,
          autoGainControl: true,
        },
      })

      microphoneStreamRef.current = stream

      // Create audio context if it doesn't exist
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      }

      // Create analyser if it doesn't exist
      if (!analyserRef.current) {
        analyserRef.current = audioContextRef.current.createAnalyser()
        analyserRef.current.fftSize = 256 // Smaller size for better performance
        analyserRef.current.smoothingTimeConstant = 0.8 // Smoother transitions
      }

      // Create source from microphone
      const source = audioContextRef.current.createMediaStreamSource(stream)

      // Connect source to analyzer
      source.connect(analyserRef.current)

      // Make analyser available globally
      window.microphoneAnalyser = analyserRef.current

      // Apply noise reduction if enabled
      if (speechSettings.noiseReduction) {
        // In a real implementation, you would apply a noise reduction filter here
        // This is a simplified example
        const filter = audioContextRef.current.createBiquadFilter()
        filter.type = "lowpass"
        filter.frequency.value = 8000

        source.connect(filter)
        filter.connect(analyserRef.current)
      }

      console.log("Advanced audio processing set up successfully")
      return true
    } catch (err) {
      console.error("Error setting up audio processing:", err)
      setError(`Failed to access microphone. Please check your permissions: ${err.message}`)
      return false
    }
  }

  // Start voice recognition
  const startListening = async () => {
    if (!recognitionRef.current) {
      setError("Speech recognition is not supported in your browser. Please try a different browser like Chrome.")
      return
    }

    setStatus("listening")
    setIsListening(true)
    setTranscript("")
    setError("")

    // Set up advanced audio processing
    const audioSetupSuccess = await setupAudioProcessing()
    if (!audioSetupSuccess) {
      stopListening()
      return
    }

    // Set language based on user preference
    if (language === "auto") {
      recognitionRef.current.lang = "en-US" // Start with English, will detect language later
    } else {
      recognitionRef.current.lang = language
    }

    // Configure recognition based on settings
    recognitionRef.current.continuous = true
    recognitionRef.current.interimResults = true

    // Adjust sensitivity
    // Note: Web Speech API doesn't directly expose sensitivity settings,
    // but we can simulate it by adjusting how we handle results
    const sensitivityThreshold = 1 - speechSettings.sensitivity

    recognitionRef.current.onstart = () => {
      setStatus("listening")
      console.log("Speech recognition started")
    }

    recognitionRef.current.onresult = (event) => {
      let interimTranscript = ""
      let finalTranscript = ""

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript
        const confidence = event.results[i][0].confidence

        // Apply sensitivity threshold
        if (confidence > sensitivityThreshold) {
          if (event.results[i].isFinal) {
            finalTranscript += transcript

            // Detect language if set to auto
            if (language === "auto" && !detectedLanguage) {
              detectLanguage(transcript)
            }

            // Process the query when the user stops speaking
            processQuery(finalTranscript)
          } else {
            interimTranscript += transcript
          }
        }
      }

      setTranscript(finalTranscript || interimTranscript)
    }

    recognitionRef.current.onerror = (event) => {
      console.error("Speech recognition error", event.error)

      let errorMessage = `Speech recognition error: ${event.error}`

      // Provide more helpful error messages
      if (event.error === "no-speech") {
        errorMessage = "No speech was detected. Please make sure your microphone is working and try speaking louder."
      } else if (event.error === "audio-capture") {
        errorMessage =
          "No microphone was found. Please ensure your microphone is connected and permissions are granted."
      } else if (event.error === "not-allowed") {
        errorMessage = "Microphone access was denied. Please allow microphone access in your browser settings."
      } else if (event.error === "network") {
        errorMessage = "Network error occurred. Please check your internet connection."
      } else if (event.error === "aborted") {
        errorMessage = "Speech recognition was aborted."
      }

      setError(errorMessage)
      setStatus("error")
      setIsListening(false)
    }

    recognitionRef.current.onend = () => {
      console.log("Speech recognition ended")
      // Restart listening if we're still in listening mode
      if (status === "listening" && isListening) {
        console.log("Restarting speech recognition")
        recognitionRef.current?.start()
      } else {
        setIsListening(false)
      }
    }

    try {
      recognitionRef.current.start()
      console.log("Speech recognition requested to start")
    } catch (err) {
      console.error("Error starting speech recognition:", err)
      setError(`Failed to start speech recognition: ${err.message}`)
      setStatus("error")
      setIsListening(false)
    }
  }

  // Stop listening
  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }

    if (microphoneStreamRef.current) {
      microphoneStreamRef.current.getTracks().forEach((track) => track.stop())
    }

    setIsListening(false)
  }

  // Process the voice query
  const processQuery = async (query: string) => {
    if (!excelData || !nlpModel.loaded) return

    setStatus("processing")

    try {
      // Use the current detected language or fallback to the selected language
      const queryLanguage = detectedLanguage || (language === "auto" ? "en-US" : language)

      // Process query with local AI
      const result = await processQueryWithLocalAI(query, excelData, queryLanguage, nlpModel)

      // Update state with response
      setResponse(result)

      // Add to query history
      setQueryHistory((prev) => [...prev, { query, response: result }])

      // Speak the response
      speakResponse(result.answer)

      // Set active tab to response
      setActiveTab("response")
    } catch (err) {
      console.error("Query processing error:", err)

      const errorMsg =
        detectedLanguage === "de-DE" || language === "de-DE"
          ? "Es gab einen Fehler bei der Verarbeitung Ihrer Anfrage. Bitte versuchen Sie es erneut."
          : "There was an error processing your query. Please try again."

      setError(errorMsg)
      setStatus("error")
      speakResponse(errorMsg)
    }
  }

  // Speak the response
  const speakResponse = (text: string) => {
    if (!synthRef.current) return

    setStatus("speaking")

    // Cancel any ongoing speech
    synthRef.current.cancel()

    const utterance = new SpeechSynthesisUtterance(text)

    // Set language based on detected or selected language
    utterance.lang = detectedLanguage || (language === "auto" ? "en-US" : language) || "en-US"

    // Apply speech settings
    utterance.rate = speechSettings.rate
    utterance.pitch = speechSettings.pitch
    utterance.volume = speechSettings.volume

    utterance.onend = () => {
      setStatus("listening")
    }

    utterance.onerror = () => {
      setError("Error speaking response")
      setStatus("error")
    }

    synthRef.current.speak(utterance)
  }

  // Handle follow-up question
  const handleFollowUpQuestion = (question: string) => {
    setTranscript(question)
    processQuery(question)
  }

  // Reset the application
  const handleReset = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }

    if (synthRef.current) {
      synthRef.current.cancel()
    }

    if (microphoneStreamRef.current) {
      microphoneStreamRef.current.getTracks().forEach((track) => track.stop())
    }

    setStatus("idle")
    setFile(null)
    setExcelData(null)
    setTranscript("")
    setResponse(null)
    setError("")
    setUploadProgress(0)
    setDetectedLanguage(null)
    setIsListening(false)
    setQueryHistory([])
  }

  // Update speech settings
  const updateSpeechSettings = (newSettings: Partial<SpeechSettings>) => {
    setSpeechSettings((prev) => ({
      ...prev,
      ...newSettings,
    }))

    // Restart audio processing if needed
    if ("noiseReduction" in newSettings || "echoReduction" in newSettings || "multiSpeaker" in newSettings) {
      if (microphoneStreamRef.current) {
        microphoneStreamRef.current.getTracks().forEach((track) => track.stop())
      }

      if (isListening) {
        setupAudioProcessing()
      }
    }
  }

  // Get status message
  const getStatusMessage = () => {
    const lang = detectedLanguage || (language === "auto" ? "en-US" : language)

    switch (status) {
      case "idle":
        return "Upload an Excel file to begin"
      case "uploading":
        return "Uploading file..."
      case "uploaded":
        return "File uploaded successfully"
      case "loading-model":
        return "Loading AI model..."
      case "listening":
        return lang === "de-DE" ? "Höre zu..." : "Listening..."
      case "processing":
        return lang === "de-DE" ? "Verarbeite Anfrage..." : "Processing query..."
      case "speaking":
        return lang === "de-DE" ? "Spreche..." : "Speaking..."
      case "error":
        return error || "An error occurred"
      default:
        return ""
    }
  }

  // Get status color
  const getStatusColor = () => {
    switch (status) {
      case "idle":
        return "bg-muted"
      case "uploading":
        return "bg-amber-500"
      case "uploaded":
        return "bg-green-500"
      case "loading-model":
        return "bg-purple-500"
      case "listening":
        return "bg-blue-500 animate-pulse"
      case "processing":
        return "bg-purple-500"
      case "speaking":
        return "bg-teal-500 animate-pulse"
      case "error":
        return "bg-red-500"
      default:
        return "bg-muted"
    }
  }

  // Render data visualization
  const renderVisualization = () => {
    if (!response || !response.data) return null

    switch (visualization) {
      case "table":
        return (
          <div className="overflow-auto max-h-[300px]">
            <Table>
              <TableHeader>
                <TableRow>
                  {response.data[0] &&
                    Object.keys(response.data[0]).map((key, i) => <TableHead key={i}>{key}</TableHead>)}
                </TableRow>
              </TableHeader>
              <TableBody>
                {response.data.map((row, i) => (
                  <TableRow key={i}>
                    {Object.values(row).map((value, j) => (
                      <TableCell key={j}>{String(value)}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )
      case "chart":
        return (
          <div className="h-[300px] flex items-center justify-center">
            <div className="text-center text-muted-foreground">
              <BarChart4 className="h-16 w-16 mx-auto mb-2 opacity-50" />
              <p>Chart visualization would be rendered here</p>
              <p className="text-sm">In a real implementation, this would show a chart based on the data</p>
            </div>
          </div>
        )
      default:
        return null
    }
  }

  // Get confidence level text and color
  const getConfidenceInfo = (confidence: number) => {
    let text = ""
    let color = ""

    if (confidence >= 0.8) {
      text = detectedLanguage === "de-DE" ? "Hohe Konfidenz" : "High Confidence"
      color = "text-green-500"
    } else if (confidence >= 0.5) {
      text = detectedLanguage === "de-DE" ? "Mittlere Konfidenz" : "Medium Confidence"
      color = "text-amber-500"
    } else {
      text = detectedLanguage === "de-DE" ? "Niedrige Konfidenz" : "Low Confidence"
      color = "text-red-500"
    }

    return { text, color }
  }

  // Export data to CSV
  const exportToCSV = () => {
    if (!response || !response.data) return

    // Convert data to CSV
    const headers = Object.keys(response.data[0]).join(",")
    const rows = response.data.map((row) => Object.values(row).join(","))
    const csv = [headers, ...rows].join("\n")

    // Create download link
    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "excel_analysis_result.csv"
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // Add this function to the component
  const testMicrophone = async () => {
    setStatus("listening")
    const success = await setupAudioProcessing()
    if (success) {
      setIsListening(true)
      setTimeout(() => {
        if (isListening) {
          setIsListening(false)
          setStatus("uploaded")
        }
      }, 5000) // Test for 5 seconds
    } else {
      setStatus("error")
    }
  }

  // Add this function to handle manual query submission
  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (manualInput.trim()) {
      processQuery(manualInput)
      setManualInput("")
    }
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold text-center mb-2">Excel Voice Analyzer</h1>
      <p className="text-center text-muted-foreground mb-8">
        Upload Excel files and analyze them using natural voice commands with free, offline AI processing
      </p>

      <div className="max-w-4xl mx-auto">
        {/* File Upload Area */}
        {status === "idle" && (
          <Card>
            <CardHeader>
              <CardTitle>Upload Excel File</CardTitle>
              <CardDescription>
                Upload your Excel file (.xlsx) to analyze it using voice commands with AI assistance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                ref={dropZoneRef}
                className="border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors hover:border-primary"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg font-semibold mb-2">Drag & Drop or Click to Upload</h3>
                <p className="text-sm text-muted-foreground mb-4">Only .xlsx files are supported</p>
                <Button variant="outline">Select File</Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".xlsx"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileUpload(e.target.files[0])
                    }
                  }}
                />
              </div>

              <div className="mt-6 space-y-4">
                <div className="flex justify-between items-center">
                  <Label htmlFor="language">Voice Recognition Language</Label>
                  <Select value={language || "auto"} onValueChange={(value) => setLanguage(value as Language)}>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Select Language" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">
                        <div className="flex items-center gap-2">
                          <Globe className="h-4 w-4" />
                          <span>Auto Detect</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="en-US">English</SelectItem>
                      <SelectItem value="de-DE">German</SelectItem>
                    </SelectContent>
                  </div>

                <Separator />

                <div className="flex justify-between items-center">
                  <Label>Advanced Voice Settings</Label>
                  <Button variant="ghost" size="sm" onClick={() => setShowSettings(!showSettings)}>
                    {showSettings ? "Hide Settings" : "Show Settings"}
                  </Button>
                </div>

                {showSettings && (
                  <div className="space-y-4 p-4 bg-muted/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="noise-reduction">Noise Reduction</Label>
                      <Switch
                        id="noise-reduction"
                        checked={speechSettings.noiseReduction}
                        onCheckedChange={(checked) => updateSpeechSettings({ noiseReduction: checked })}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label htmlFor="echo-reduction">Echo Reduction</Label>
                      <Switch
                        id="echo-reduction"
                        checked={speechSettings.echoReduction}
                        onCheckedChange={(checked) => updateSpeechSettings({ echoReduction: checked })}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label htmlFor="multi-speaker">Multi-Speaker Detection</Label>
                      <Switch
                        id="multi-speaker"
                        checked={speechSettings.multiSpeaker}
                        onCheckedChange={(checked) => updateSpeechSettings({ multiSpeaker: checked })}
                      />
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label htmlFor="sensitivity">Microphone Sensitivity</Label>
                        <span className="text-sm text-muted-foreground">
                          {Math.round(speechSettings.sensitivity * 100)}%
                        </span>
                      </div>
                      <Slider
                        id="sensitivity"
                        min={0.1}
                        max={1}
                        step={0.05}
                        value={[speechSettings.sensitivity]}
                        onValueChange={([value]) => updateSpeechSettings({ sensitivity: value })}
                      />
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label htmlFor="rate">Speech Rate</Label>
                        <span className="text-sm text-muted-foreground">{speechSettings.rate.toFixed(1)}x</span>
                      </div>
                      <Slider
                        id="rate"
                        min={0.5}
                        max={2}
                        step={0.1}
                        value={[speechSettings.rate]}
                        onValueChange={([value]) => updateSpeechSettings({ rate: value })}
                      />
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label htmlFor="pitch">Speech Pitch</Label>
                        <span className="text-sm text-muted-foreground">{speechSettings.pitch.toFixed(1)}</span>
                      </div>
                      <Slider
                        id="pitch"
                        min={0.5}
                        max={2}
                        step={0.1}
                        value={[speechSettings.pitch]}
                        onValueChange={([value]) => updateSpeechSettings({ pitch: value })}
                      />
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label htmlFor="volume">Speech Volume</Label>
                        <span className="text-sm text-muted-foreground">
                          {Math.round(speechSettings.volume * 100)}%
                        </span>
                      </div>
                      <Slider
                        id="volume"
                        min={0.1}
                        max={1}
                        step={0.05}
                        value={[speechSettings.volume]}
                        onValueChange={([value]) => updateSpeechSettings({ volume: value })}
                      />
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Upload Progress */}
        {status === "uploading" && (
          <Card>
            <CardHeader>
              <CardTitle>Uploading File</CardTitle>
              <CardDescription>{file?.name}</CardDescription>
            </CardHeader>
            <CardContent>
              <Progress value={uploadProgress} className="mb-2" />
              <p className="text-sm text-center text-muted-foreground">{uploadProgress}% complete</p>
            </CardContent>
          </Card>
        )
}

{
  /* Loading Model */
}
{
  status === "loading-model" && (
    <Card>
      <CardHeader>
        <CardTitle>Loading AI Model</CardTitle>
        <CardDescription>Preparing the offline AI model for voice analysis</CardDescription>
      </CardHeader>
      <CardContent className="text-center py-8">
        <div className="flex flex-col items-center justify-center gap-4">
          <Brain className="h-16 w-16 text-primary animate-pulse" />
          <Progress value={modelLoading ? 70 : 100} className="w-64" />
          <p className="text-sm text-muted-foreground">
            Loading TensorFlow.js model for natural language processing...
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

{
  /* Analysis Interface */
}
{
  status !== "idle" && status !== "uploading" && status !== "loading-model" && (
    <div className="space-y-6">
      {/* Status Bar */}
      <div className="flex items-center justify-between bg-muted p-4 rounded-lg">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
          <span className="font-medium">{getStatusMessage()}</span>
        </div>

        <div className="flex items-center gap-2">
          {file && (
            <Badge variant="outline" className="flex items-center gap-1">
              <FileSpreadsheet className="h-3 w-3" />
              {file.name}
            </Badge>
          )}

          {detectedLanguage && <Badge variant="secondary">{detectedLanguage === "de-DE" ? "German" : "English"}</Badge>}

          <Button
            variant={isListening ? "destructive" : "default"}
            size="sm"
            onClick={() => (isListening ? stopListening() : startListening())}
          >
            {isListening ? "Stop Listening" : "Start Listening"}
          </Button>

          <Button variant="ghost" size="sm" onClick={handleReset}>
            Reset
          </Button>
        </div>
      </div>

      {/* File Info */}
      {excelData && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Excel Data</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm">
              <p>
                <strong>Active Sheet:</strong> {excelData.activeSheet}
              </p>
              <p>
                <strong>Rows:</strong> {excelData.sheets[excelData.activeSheet]?.length || 0}
              </p>
              <p>
                <strong>Columns:</strong>{" "}
                {excelData.sheets[excelData.activeSheet]?.[0]
                  ? Object.keys(excelData.sheets[excelData.activeSheet][0]).length
                  : 0}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Voice Input/Output */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <CardTitle className="text-lg">Voice Interaction</CardTitle>
            {status === "listening" && (
              <div className="flex items-center gap-2">
                <Mic className="h-4 w-4 text-blue-500 animate-pulse" />
                <span className="text-sm text-muted-foreground">
                  {detectedLanguage === "de-DE" ? "Sprechen Sie jetzt..." : "Speak now..."}
                </span>
              </div>
            )}
            {status === "speaking" && (
              <div className="flex items-center gap-2">
                <Volume2 className="h-4 w-4 text-teal-500 animate-pulse" />
                <span className="text-sm text-muted-foreground">
                  {detectedLanguage === "de-DE" ? "Sprechend..." : "Speaking..."}
                </span>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Transcript */}
            <div className="bg-muted p-3 rounded-md min-h-[60px]">
              <div className="flex justify-between items-center mb-1">
                <p className="text-sm font-medium">{detectedLanguage === "de-DE" ? "Ihre Anfrage:" : "Your Query:"}</p>
                {isListening && <MicrophoneLevelIndicator isListening={isListening} />}
              </div>
              <p className="text-sm">
                {transcript || (
                  <span className="text-muted-foreground italic">
                    {detectedLanguage === "de-DE"
                      ? "Ihre Sprache wird hier angezeigt..."
                      : "Your speech will appear here..."}
                  </span>
                )}
              </p>
            </div>

            {/* Manual Input Option */}
            <form onSubmit={handleManualSubmit} className="flex gap-2">
              <input
                type="text"
                value={manualInput}
                onChange={(e) => setManualInput(e.target.value)}
                placeholder={
                  detectedLanguage === "de-DE"
                    ? "Oder geben Sie Ihre Anfrage hier ein..."
                    : "Or type your query here..."
                }
                className="flex-1 px-3 py-2 text-sm rounded-md border border-input bg-background"
              />
              <Button type="submit" size="sm" disabled={status === "processing"}>
                {detectedLanguage === "de-DE" ? "Senden" : "Submit"}
              </Button>
            </form>

            {/* Microphone Test Button */}
            {!isListening && status !== "processing" && (
              <div className="flex justify-center">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={testMicrophone}
                  className="text-xs flex items-center gap-1"
                >
                  <Mic className="h-3 w-3" />
                  {detectedLanguage === "de-DE" ? "Mikrofon testen" : "Test Microphone"}
                </Button>
              </div>
            )}

            {/* Troubleshooting Tips */}
            {error && error.includes("microphone") && (
              <div className="text-xs text-muted-foreground space-y-1 p-2 bg-muted/50 rounded-md">
                <p className="font-medium">{detectedLanguage === "de-DE" ? "Fehlerbehebung:" : "Troubleshooting:"}</p>
                <ul className="list-disc pl-4 space-y-1">
                  <li>
                    {detectedLanguage === "de-DE"
                      ? "Stellen Sie sicher, dass Ihr Mikrofon angeschlossen ist"
                      : "Make sure your microphone is connected"}
                  </li>
                  <li>
                    {detectedLanguage === "de-DE"
                      ? "Erlauben Sie dem Browser Zugriff auf Ihr Mikrofon"
                      : "Allow browser access to your microphone"}
                  </li>
                  <li>
                    {detectedLanguage === "de-DE"
                      ? "Versuchen Sie, einen anderen Browser zu verwenden"
                      : "Try using a different browser"}
                  </li>
                  <li>
                    {detectedLanguage === "de-DE"
                      ? "Verwenden Sie die manuelle Eingabe oben"
                      : "Use the manual input option above"}
                  </li>
                </ul>
              </div>
            )}

            {/* Response Tabs */}
            {response && (
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid grid-cols-3">
                  <TabsTrigger value="response">Response</TabsTrigger>
                  <TabsTrigger value="explanation">Explanation</TabsTrigger>
                  <TabsTrigger value="data">Data</TabsTrigger>
                </TabsList>

                <TabsContent value="response" className="space-y-4">
                  {/* AI Confidence */}
                  {response.confidence > 0 && (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Brain className="h-4 w-4 text-purple-500" />
                        <span className="text-xs font-medium">AI Analysis</span>
                      </div>
                      <Badge variant="outline" className={`${getConfidenceInfo(response.confidence).color}`}>
                        {getConfidenceInfo(response.confidence).text}
                      </Badge>
                    </div>
                  )}

                  {/* Response */}
                  <div className="bg-primary/10 p-4 rounded-md">
                    <p className="text-sm font-medium mb-2">
                      {detectedLanguage === "de-DE" ? "Antwort:" : "Response:"}
                    </p>
                    <p className="text-sm">{response.answer}</p>
                  </div>

                  {/* Follow-up Questions */}
                  {response.followUpQuestions && response.followUpQuestions.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-muted-foreground">
                        {detectedLanguage === "de-DE" ? "Mögliche Folgefragen:" : "Suggested Follow-up Questions:"}
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {response.followUpQuestions.map((question, index) => (
                          <Button
                            key={index}
                            variant="outline"
                            size="sm"
                            onClick={() => handleFollowUpQuestion(question)}
                            className="text-xs"
                          >
                            {question}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="explanation">
                  <div className="bg-muted p-4 rounded-md">
                    <div className="flex items-center gap-2 mb-2">
                      <Wand2 className="h-4 w-4 text-amber-500" />
                      <p className="text-sm font-medium">
                        {detectedLanguage === "de-DE" ? "Erklärung:" : "Explanation:"}
                      </p>
                    </div>
                    <p className="text-sm">
                      {response.explanation || (
                        <span className="text-muted-foreground italic">
                          {detectedLanguage === "de-DE" ? "Keine Erklärung verfügbar." : "No explanation available."}
                        </span>
                      )}
                    </p>

                    {response.usedColumns && response.usedColumns.length > 0 && (
                      <div className="mt-4">
                        <p className="text-xs font-medium text-muted-foreground mb-1">
                          {detectedLanguage === "de-DE" ? "Verwendete Spalten:" : "Columns Used:"}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {response.usedColumns.map((col, index) => (
                            <Badge key={index} variant="outline" className="text-xs">
                              {col}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {response.operation && (
                      <div className="mt-4">
                        <p className="text-xs font-medium text-muted-foreground mb-1">
                          {detectedLanguage === "de-DE" ? "Operation:" : "Operation:"}
                        </p>
                        <Badge variant="secondary" className="text-xs">
                          {response.operation}
                        </Badge>
                      </div>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="data">
                  {response.data && response.data.length > 0 ? (
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <p className="text-sm font-medium">
                          {detectedLanguage === "de-DE" ? "Datenvisualisierung:" : "Data Visualization:"}
                        </p>
                        <div className="flex items-center gap-2">
                          <Button
                            variant={visualization === "table" ? "default" : "outline"}
                            size="sm"
                            onClick={() => setVisualization("table")}
                            className="text-xs"
                          >
                            Table
                          </Button>
                          <Button
                            variant={visualization === "chart" ? "default" : "outline"}
                            size="sm"
                            onClick={() => setVisualization("chart")}
                            className="text-xs"
                          >
                            Chart
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={exportToCSV}
                            className="text-xs flex items-center gap-1"
                          >
                            <Download className="h-3 w-3" />
                            Export
                          </Button>
                        </div>
                      </div>

                      {renderVisualization()}
                    </div>
                  ) : (
                    <div className="bg-muted p-4 rounded-md text-center">
                      <p className="text-sm text-muted-foreground">
                        {detectedLanguage === "de-DE"
                          ? "Keine Daten für diese Anfrage verfügbar."
                          : "No data available for this query."}
                      </p>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </div>
        </CardContent>
        <CardFooter className="flex flex-col space-y-2">
          <p className="text-xs text-muted-foreground">
            {detectedLanguage === "de-DE"
              ? 'Stellen Sie Fragen zu Ihren Excel-Daten, z.B. "Was ist der Durchschnitt von Spalte X?" oder "Ist ID 48 in den Daten?"'
              : 'Ask questions about your Excel data, e.g. "What is the average of column X?" or "Is ID 48 in the data?"'}
          </p>
          <div className="text-xs text-muted-foreground pt-1 border-t border-border">
            <p className="font-medium mb-1">
              {detectedLanguage === "de-DE"
                ? "Tipps für bessere Spracherkennung:"
                : "Tips for better voice recognition:"}
            </p>
            <ul className="list-disc pl-4 space-y-1">
              <li>
                {detectedLanguage === "de-DE"
                  ? "Sprechen Sie deutlich und nicht zu schnell"
                  : "Speak clearly and not too fast"}
              </li>
              <li>
                {detectedLanguage === "de-DE" ? "Reduzieren Sie Hintergrundgeräusche" : "Reduce background noise"}
              </li>
              <li>
                {detectedLanguage === "de-DE"
                  ? "Verwenden Sie ein hochwertiges Mikrofon"
                  : "Use a good quality microphone"}
              </li>
              <li>
                {detectedLanguage === "de-DE"
                  ? "Wenn die Spracherkennung nicht funktioniert, nutzen Sie die Texteingabe"
                  : "If voice recognition isn't working, use the text input"}
              </li>
            </ul>
          </div>
        </CardFooter>
      </Card>

      {/* Query History */}
      {queryHistory.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Query History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
              {queryHistory.map((item, index) => (
                <div
                  key={index}
                  className="p-2 rounded-md hover:bg-muted cursor-pointer text-sm"
                  onClick={() => {
                    setTranscript(item.query)
                    setResponse(item.response)
                    setActiveTab("response")
                  }}
                >
                  <p className="font-medium truncate">{item.query}</p>
                  <p className="text-xs text-muted-foreground truncate">{item.response.answer}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error Message */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  )
}
