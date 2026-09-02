import { useEffect, useState } from 'react'
import {
  RiBarChartGroupedLine,
  RiCheckboxCircleLine,
  RiErrorWarningLine,
  RiHourglassLine,
  RiLineChartLine,
  RiMicLine,
  RiPieChartLine,
  RiPulseLine,
  RiRefreshLine,
  RiSparklingLine,
  RiTimeLine,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Spinner } from '@/components/ui/spinner'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { api } from '@/services/api'
import type { AnalyticsReport } from '@/types'

function formatSeconds(sec?: number | null) {
  if (sec === null || sec === undefined) return '--'
  return `${sec.toFixed(1)}s`
}

function formatPercent(fraction?: number | null) {
  if (fraction === null || fraction === undefined) return '--'
  return `${(fraction * 100).toFixed(1)}%`
}

export function AnalyticsView() {
  const [report, setReport] = useState<AnalyticsReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadReport = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getReport()
      setReport(data)
    } catch (err: any) {
      setError(err.message || 'Failed to load analytics report')
      toast.error('Failed to load analytics')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadReport()
  }, [])

  if (loading && !report) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-sm text-muted-foreground">
        <Spinner className="size-6 text-primary" />
        <span>Loading analytics &amp; quality metrics…</span>
      </div>
    )
  }

  if (error && !report) {
    return (
      <div className="p-6 max-w-2xl mx-auto">
        <Alert variant="destructive">
          <RiErrorWarningLine className="size-5" />
          <AlertTitle>Could not load analytics</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button onClick={loadReport} className="mt-4 gap-2">
          <RiRefreshLine className="size-4" /> Retry
        </Button>
      </div>
    )
  }

  if (!report) return null

  const corpus = report.corpus
  const labels = report.labels
  const throughput = report.throughput
  const queue = report.queue
  const scores = report.scores
  const splitBalance = report.split_balance
  const wordCoverage = report.word_timestamp_coverage

  const accepted = labels.accepted_unchanged || 0
  const edited = labels.edited || 0
  const unusable = labels.unusable_audio || 0
  const uncertain = labels.uncertain || 0
  const totalLabeled = labels.total || 1

  const acceptedPct = Math.round((accepted / totalLabeled) * 100)
  const editedPct = Math.round((edited / totalLabeled) * 100)
  const unusablePct = Math.round((unusable / totalLabeled) * 100)
  const uncertainPct = Math.round((uncertain / totalLabeled) * 100)

  return (
    <div className="scrollbar-thin flex-1 overflow-y-auto bg-background p-6 space-y-6">
      {/* Header & Title */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b pb-4">
        <div>
          <div className="flex items-center gap-2">
            <RiLineChartLine className="size-6 text-primary" />
            <h1 className="font-heading text-xl font-bold tracking-tight">Analytics &amp; Statistics</h1>
            <Badge variant="secondary" className="font-mono text-xs">
              Live
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            Corpus size, annotation throughput velocity, model agreement, and pipeline health.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="font-mono text-[11px] text-muted-foreground">
            Updated: {new Date(report.generated_at).toLocaleTimeString()}
          </span>
          <Button variant="outline" size="sm" onClick={loadReport} className="h-8 gap-1.5">
            <RiRefreshLine className="size-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      {/* TOP KPI HERO CARDS */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Audio Hours */}
        <Card className="bg-card/50">
          <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
            <CardTitle className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Corpus Audio
            </CardTitle>
            <RiMicLine className="size-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="font-mono text-2xl font-bold">{corpus.audio_hours.toFixed(1)} hrs</div>
            <p className="mt-1 text-xs text-muted-foreground">
              {corpus.segments.toLocaleString()} segments across {corpus.episodes} episodes
            </p>
          </CardContent>
        </Card>

        {/* Accept Rate */}
        <Card className="bg-card/50">
          <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
            <CardTitle className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Accept Rate
            </CardTitle>
            <RiCheckboxCircleLine className="size-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            <div className="font-mono text-2xl font-bold text-emerald-600 dark:text-emerald-400">
              {formatPercent(report.accept_rate)}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {labels.total.toLocaleString()} total human annotations submitted
            </p>
          </CardContent>
        </Card>

        {/* Throughput */}
        <Card className="bg-card/50">
          <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
            <CardTitle className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Velocity
            </CardTitle>
            <RiPulseLine className="size-4 text-warning" />
          </CardHeader>
          <CardContent>
            <div className="font-mono text-2xl font-bold">
              {formatSeconds(throughput.median_seconds_per_segment)}
              <span className="text-xs font-normal text-muted-foreground ml-1">/ seg</span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {throughput.segments_per_hour
                ? `${throughput.segments_per_hour.toFixed(0)} segs / hr`
                : '--'}{' '}
              &middot; {throughput.annotator_hours.toFixed(1)} human hours logged
            </p>
          </CardContent>
        </Card>

        {/* Backlog & Remaining */}
        <Card className="bg-card/50">
          <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
            <CardTitle className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Backlog Left
            </CardTitle>
            <RiHourglassLine className="size-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="font-mono text-2xl font-bold">{queue.backlog.toLocaleString()} segs</div>
            <p className="mt-1 text-xs text-muted-foreground">
              Est.{' '}
              {queue.projected_hours_to_finish
                ? `${queue.projected_hours_to_finish.toFixed(1)} hrs left`
                : '--'}{' '}
              to complete corpus
            </p>
          </CardContent>
        </Card>
      </div>

      {/* SECTION: Labels Disposition & Split Balance */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Disposition Mix */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <RiPieChartLine className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">Annotation Disposition Mix</CardTitle>
            </div>
            <CardDescription className="text-xs">
              Breakdown of decisions made by annotators on the corpus.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Visual Multi-color Progress Bar */}
            <div className="flex h-3 w-full overflow-hidden rounded-full bg-muted">
              <div
                style={{ width: `${acceptedPct}%` }}
                className="bg-emerald-500"
                title={`Accepted Unchanged: ${acceptedPct}%`}
              />
              <div
                style={{ width: `${editedPct}%` }}
                className="bg-blue-500"
                title={`Edited: ${editedPct}%`}
              />
              <div
                style={{ width: `${unusablePct}%` }}
                className="bg-rose-500"
                title={`Unusable Audio: ${unusablePct}%`}
              />
              <div
                style={{ width: `${uncertainPct}%` }}
                className="bg-amber-500"
                title={`Uncertain: ${uncertainPct}%`}
              />
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
              <div className="rounded-md border p-2.5 space-y-1">
                <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-medium">
                  <span className="size-2 rounded-full bg-emerald-500" />
                  Accepted
                </div>
                <div className="font-mono text-lg font-bold">{accepted.toLocaleString()}</div>
                <div className="text-[10px] text-muted-foreground">{acceptedPct}% of labels</div>
              </div>

              <div className="rounded-md border p-2.5 space-y-1">
                <div className="flex items-center gap-1.5 text-blue-600 dark:text-blue-400 font-medium">
                  <span className="size-2 rounded-full bg-blue-500" />
                  Edited
                </div>
                <div className="font-mono text-lg font-bold">{edited.toLocaleString()}</div>
                <div className="text-[10px] text-muted-foreground">{editedPct}% of labels</div>
              </div>

              <div className="rounded-md border p-2.5 space-y-1">
                <div className="flex items-center gap-1.5 text-rose-600 dark:text-rose-400 font-medium">
                  <span className="size-2 rounded-full bg-rose-500" />
                  Unusable
                </div>
                <div className="font-mono text-lg font-bold">{unusable.toLocaleString()}</div>
                <div className="text-[10px] text-muted-foreground">{unusablePct}% of labels</div>
              </div>

              <div className="rounded-md border p-2.5 space-y-1">
                <div className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400 font-medium">
                  <span className="size-2 rounded-full bg-amber-500" />
                  Uncertain
                </div>
                <div className="font-mono text-lg font-bold">{uncertain.toLocaleString()}</div>
                <div className="text-[10px] text-muted-foreground">{uncertainPct}% of labels</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Split Balance */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <RiBarChartGroupedLine className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">Corpus Split Balance</CardTitle>
            </div>
            <CardDescription className="text-xs">
              Deterministic episode-hashed distribution frozen at import.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs font-semibold">Split</TableHead>
                  <TableHead className="text-xs font-semibold text-right">Episodes</TableHead>
                  <TableHead className="text-xs font-semibold text-right">Segments</TableHead>
                  <TableHead className="text-xs font-semibold text-right">Audio (hrs)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(splitBalance).map(([splitName, entry]) => (
                  <TableRow key={splitName}>
                    <TableCell className="font-mono text-xs uppercase font-medium">
                      {splitName}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {entry.episodes}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {entry.segments.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs font-semibold">
                      {entry.hours.toFixed(2)}h
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      {/* SECTION: Upstream Model Quality & Agreement */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <RiSparklingLine className="size-4 text-primary" />
            <CardTitle className="text-sm font-semibold">
              Quality &amp; Agreement Metrics
            </CardTitle>
          </div>
          <CardDescription className="text-xs">
            Model disagreement rates, language switching density, and script conflict averages.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {/* Code Switch Density */}
            <div className="rounded-lg border p-4 space-y-1">
              <span className="font-heading text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
                Mean CMI (Code Switch)
              </span>
              <div className="font-mono text-2xl font-bold">
                {scores.mean_code_switch_density !== null
                  ? `${(scores.mean_code_switch_density * 100).toFixed(1)}%`
                  : '--'}
              </div>
              <p className="text-[11px] text-muted-foreground">
                Density of intra-sentential English-Nepali code-switching.
              </p>
            </div>

            {/* Word Disagreement Rate */}
            <div className="rounded-lg border p-4 space-y-1">
              <span className="font-heading text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
                Word Disagreement
              </span>
              <div className="font-mono text-2xl font-bold">
                {scores.mean_word_disagreement_rate !== null
                  ? `${(scores.mean_word_disagreement_rate * 100).toFixed(1)}%`
                  : '--'}
              </div>
              <p className="text-[11px] text-muted-foreground">
                Token-level divergence between ASR candidate hypotheses.
              </p>
            </div>

            {/* Script Conflict Rate */}
            <div className="rounded-lg border p-4 space-y-1">
              <span className="font-heading text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
                Script Conflict
              </span>
              <div className="font-mono text-2xl font-bold">
                {scores.mean_script_conflict_rate !== null
                  ? `${(scores.mean_script_conflict_rate * 100).toFixed(1)}%`
                  : '--'}
              </div>
              <p className="text-[11px] text-muted-foreground">
                Devanagari vs Latin alphabet script conflict rate.
              </p>
            </div>

            {/* Word Timestamps Coverage */}
            <div className="rounded-lg border p-4 space-y-1">
              <span className="font-heading text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
                Word Timestamps
              </span>
              <div className="font-mono text-2xl font-bold">
                {formatPercent(wordCoverage.fraction)}
              </div>
              <p className="text-[11px] text-muted-foreground">
                {wordCoverage.hypotheses_with_words} / {wordCoverage.hypotheses_total} hypotheses with
                word-level timing.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* SECTION: Pipeline Health - Daily Accept Rate Trend */}
      {report.accept_rate_by_day && report.accept_rate_by_day.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <RiTimeLine className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">
                Daily Pipeline Health &amp; Accept Rate Trend
              </CardTitle>
            </div>
            <CardDescription className="text-xs">
              Daily accept rate monitoring detects changes or drift in upstream models over time.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs font-semibold">Date</TableHead>
                    <TableHead className="text-xs font-semibold text-right">Labeled</TableHead>
                    <TableHead className="text-xs font-semibold text-right">Accepted Unchanged</TableHead>
                    <TableHead className="text-xs font-semibold text-right">Accept Rate</TableHead>
                    <TableHead className="text-xs font-semibold w-48">Health Indicator</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {report.accept_rate_by_day.map((row) => {
                    const pct =
                      row.accept_rate !== null ? Math.round(row.accept_rate * 100) : 0

                    return (
                      <TableRow key={row.day}>
                        <TableCell className="font-mono text-xs">{row.day}</TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {row.labeled}
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {row.accepted}
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs font-bold text-foreground">
                          {formatPercent(row.accept_rate)}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Progress value={pct} className="h-2 flex-1" />
                            <span className="font-mono text-[10px] text-muted-foreground w-8 text-right">
                              {pct}%
                            </span>
                          </div>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
