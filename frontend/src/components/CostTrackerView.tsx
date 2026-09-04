import { useCallback, useEffect, useState } from 'react'
import {
  RiDatabase2Line,
  RiErrorWarningLine,
  RiInformationLine,
  RiMoneyDollarCircleLine,
  RiPulseLine,
  RiRefreshLine,
  RiSearchLine,
  RiServerLine,
  RiTimeLine,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
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
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type {
  CostReportResponse,
  CostRequestsResponse,
  LlmRequestItem,
  VendorCostBreakdown,
} from '@/types'

function formatCostUsd(cost?: number | null, precision = 4): string {
  if (cost === null || cost === undefined) return '$0.00'
  if (cost === 0) return '$0.0000'
  if (cost < 0.001) return `$${cost.toFixed(6)}`
  return `$${cost.toFixed(precision)}`
}

function formatLatency(ms?: number | null): string {
  if (ms === null || ms === undefined) return '--'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function vendorColor(vendor: string) {
  if (vendor.includes('ElevenLabs')) {
    return {
      text: 'text-indigo-500 dark:text-indigo-400',
      bg: 'bg-indigo-500/10 dark:bg-indigo-500/20',
      border: 'border-indigo-500/30',
      bar: 'bg-indigo-500',
    }
  }
  if (vendor.includes('OpenRouter')) {
    return {
      text: 'text-sky-500 dark:text-sky-400',
      bg: 'bg-sky-500/10 dark:bg-sky-500/20',
      border: 'border-sky-500/30',
      bar: 'bg-sky-500',
    }
  }
  if (vendor.includes('Vertex') || vendor.includes('Google')) {
    return {
      text: 'text-emerald-500 dark:text-emerald-400',
      bg: 'bg-emerald-500/10 dark:bg-emerald-500/20',
      border: 'border-emerald-500/30',
      bar: 'bg-emerald-500',
    }
  }
  return {
    text: 'text-amber-500 dark:text-amber-400',
    bg: 'bg-amber-500/10 dark:bg-amber-500/20',
    border: 'border-amber-500/30',
    bar: 'bg-amber-500',
  }
}

export function CostTrackerView() {
  const [report, setReport] = useState<CostReportResponse | null>(null)
  const [loadingReport, setLoadingReport] = useState(true)
  const [reportError, setReportError] = useState<string | null>(null)

  // Requests ledger state
  const [requests, setRequests] = useState<CostRequestsResponse | null>(null)
  const [loadingRequests, setLoadingRequests] = useState(false)
  const [selectedVendor, setSelectedVendor] = useState<string>('all')
  const [selectedStatus, setSelectedStatus] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [page, setPage] = useState(0)
  const [expandedRequestId, setExpandedRequestId] = useState<number | null>(null)
  const pageSize = 20

  const loadReport = useCallback(async () => {
    setLoadingReport(true)
    setReportError(null)
    try {
      const data = await api.getCosts()
      setReport(data)
    } catch (err: any) {
      setReportError(err.message || 'Failed to load cost report')
      toast.error('Failed to load cost report')
    } finally {
      setLoadingReport(false)
    }
  }, [])

  const loadRequests = useCallback(async () => {
    setLoadingRequests(true)
    try {
      const data = await api.getCostRequests({
        vendor: selectedVendor !== 'all' ? selectedVendor : undefined,
        status: selectedStatus !== 'all' ? selectedStatus : undefined,
        search: searchQuery.trim() || undefined,
        limit: pageSize,
        offset: page * pageSize,
      })
      setRequests(data)
    } catch (err: any) {
      toast.error('Failed to load request ledger')
    } finally {
      setLoadingRequests(false)
    }
  }, [selectedVendor, selectedStatus, searchQuery, page])

  useEffect(() => {
    loadReport()
  }, [loadReport])

  useEffect(() => {
    loadRequests()
  }, [loadRequests])

  const handleRefreshAll = () => {
    loadReport()
    loadRequests()
    toast.success('Cost data refreshed')
  }

  if (loadingReport && !report) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-sm text-muted-foreground">
        <Spinner className="size-6 text-primary" />
        <span>Loading AI cost tracking data…</span>
      </div>
    )
  }

  if (reportError && !report) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <Alert variant="destructive">
          <RiErrorWarningLine className="size-5" />
          <AlertTitle>Could not load cost analytics</AlertTitle>
          <AlertDescription>{reportError}</AlertDescription>
        </Alert>
        <Button onClick={loadReport} className="mt-4 gap-2">
          <RiRefreshLine className="size-4" /> Retry
        </Button>
      </div>
    )
  }

  if (!report) return null

  const summary = report.summary
  const vendors = report.vendor_breakdown
  const models = report.model_breakdown
  const catalog = report.pricing_catalog
  const timeline = report.daily_timeline

  const successRate =
    summary.total_requests > 0
      ? ((summary.successful_requests / summary.total_requests) * 100).toFixed(1)
      : '100'

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 space-y-6 max-w-7xl mx-auto w-full">
      {/* Top Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight">AI Cost Tracker</h1>
            <Badge variant="outline" className="border-primary/40 bg-primary/10 text-primary text-xs font-mono">
              Live Audited
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Real-time spend and usage tracking across all 3 AI vendors (ElevenLabs, OpenRouter, and Google Cloud Vertex AI).
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefreshAll}
            disabled={loadingReport || loadingRequests}
            className="gap-1.5"
          >
            <RiRefreshLine className={cn('size-4', (loadingReport || loadingRequests) && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Hero KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Cost */}
        <Card className="relative overflow-hidden border-border/60 bg-card/60 backdrop-blur-sm shadow-sm">
          <div className="absolute top-0 left-0 h-1 w-full bg-gradient-to-r from-indigo-500 via-sky-500 to-emerald-500" />
          <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Total Incurred Cost
            </CardTitle>
            <div className="rounded-full bg-primary/10 p-1.5 text-primary">
              <RiMoneyDollarCircleLine className="size-4" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold tracking-tight text-foreground font-mono">
              {formatCostUsd(summary.total_cost_usd, 4)}
            </div>
            <p className="text-xs text-muted-foreground mt-1.5 flex items-center gap-1.5">
              <span>{summary.total_requests} total requests</span>
              <span>•</span>
              <span>{summary.dry_run_requests} dry runs</span>
            </p>
          </CardContent>
        </Card>

        {/* Success Rate */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-sm shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Request Success Rate
            </CardTitle>
            <div className="rounded-full bg-emerald-500/10 p-1.5 text-emerald-500">
              <RiPulseLine className="size-4" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold tracking-tight text-foreground font-mono">
              {successRate}%
            </div>
            <p className="text-xs text-muted-foreground mt-1.5">
              <span className="text-emerald-500 font-medium">{summary.successful_requests} succeeded</span>
              {summary.failed_requests > 0 && (
                <span className="text-destructive font-medium ml-1.5">• {summary.failed_requests} failed</span>
              )}
            </p>
          </CardContent>
        </Card>

        {/* Avg Latency */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-sm shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Avg Round-Trip Latency
            </CardTitle>
            <div className="rounded-full bg-sky-500/10 p-1.5 text-sky-500">
              <RiTimeLine className="size-4" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold tracking-tight text-foreground font-mono">
              {formatLatency(summary.average_latency_ms)}
            </div>
            <p className="text-xs text-muted-foreground mt-1.5">
              Inference + network round-trip time
            </p>
          </CardContent>
        </Card>

        {/* Token Volume */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-sm shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Token Volume (LLMs)
            </CardTitle>
            <div className="rounded-full bg-indigo-500/10 p-1.5 text-indigo-500">
              <RiDatabase2Line className="size-4" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold tracking-tight text-foreground font-mono">
              {(summary.total_prompt_tokens + summary.total_completion_tokens).toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground mt-1.5">
              {summary.total_prompt_tokens.toLocaleString()} prompt • {summary.total_completion_tokens.toLocaleString()} completion
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Vendor Breakdown Section */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold tracking-tight">Spend by AI Vendor</h2>
          <span className="text-xs text-muted-foreground">3 external providers connected</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {vendors.map((v: VendorCostBreakdown) => {
            const colors = vendorColor(v.vendor)
            return (
              <Card
                key={v.vendor}
                className={cn('border bg-card/70 backdrop-blur-sm transition-all shadow-sm hover:shadow-md', colors.border)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <span className={cn('text-xs font-semibold px-2 py-0.5 rounded-full border', colors.bg, colors.text, colors.border)}>
                      {v.vendor}
                    </span>
                    <span className="text-xs font-mono font-medium text-muted-foreground">
                      {v.percentage.toFixed(1)}% of spend
                    </span>
                  </div>
                  <CardTitle className="text-2xl font-bold font-mono pt-2">
                    {formatCostUsd(v.cost_usd, 4)}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Progress value={v.percentage} className={cn('h-1.5', colors.bar)} />
                  <div className="grid grid-cols-3 gap-2 pt-1 border-t border-border/50 text-xs">
                    <div>
                      <span className="text-muted-foreground block">Calls</span>
                      <span className="font-mono font-semibold">{v.requests}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block">Errors</span>
                      <span className={cn('font-mono font-semibold', v.failed > 0 ? 'text-destructive' : 'text-muted-foreground')}>
                        {v.failed}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block">Avg Latency</span>
                      <span className="font-mono font-semibold">{formatLatency(v.average_latency_ms)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Model Breakdown & Catalog Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Breakdown Table (2 cols on desktop) */}
        <Card className="lg:col-span-2 border-border/60 bg-card/60 shadow-sm">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base font-semibold">Configured Routes &amp; Models</CardTitle>
                <CardDescription className="text-xs">
                  Detailed cost and call volume breakdown per transcription pipeline system.
                </CardDescription>
              </div>
              <RiServerLine className="size-5 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow className="border-border/50 hover:bg-transparent">
                  <TableHead className="text-xs">Route / Model</TableHead>
                  <TableHead className="text-xs">Vendor</TableHead>
                  <TableHead className="text-xs text-right">Invocations</TableHead>
                  <TableHead className="text-xs text-right">Avg Latency</TableHead>
                  <TableHead className="text-xs text-right">Total Cost</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center py-6 text-xs text-muted-foreground">
                      No requests recorded yet.
                    </TableCell>
                  </TableRow>
                ) : (
                  models.map((m) => {
                    const colors = vendorColor(m.vendor)
                    return (
                      <TableRow key={m.route} className="border-border/40 hover:bg-muted/40">
                        <TableCell>
                          <div className="font-mono text-xs font-semibold">{m.route}</div>
                          <div className="text-[11px] text-muted-foreground truncate max-w-[200px]">
                            {m.model}
                          </div>
                        </TableCell>
                        <TableCell>
                          <span className={cn('text-[11px] font-medium px-2 py-0.5 rounded border', colors.bg, colors.text, colors.border)}>
                            {m.vendor}
                          </span>
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          <div>{m.requests}</div>
                          <div className="text-[10px] text-muted-foreground">
                            {m.successful} ok {m.failed > 0 && <span className="text-destructive">• {m.failed} err</span>}
                          </div>
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs text-muted-foreground">
                          {formatLatency(m.average_latency_ms)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs font-bold text-foreground">
                          {formatCostUsd(m.cost_usd, 4)}
                        </TableCell>
                      </TableRow>
                    )
                  })
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Pricing Catalog Reference (1 col on desktop) */}
        <Card className="border-border/60 bg-card/60 shadow-sm flex flex-col">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base font-semibold">Vendor Pricing Catalog</CardTitle>
                <CardDescription className="text-xs">
                  Official billing formulas for configured models.
                </CardDescription>
              </div>
              <RiInformationLine className="size-5 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent className="space-y-3 flex-1 overflow-y-auto max-h-[380px] pr-2">
            {catalog.map((cat) => {
              const colors = vendorColor(cat.vendor)
              return (
                <div
                  key={cat.route}
                  className="rounded-lg border border-border/50 bg-background/50 p-3 text-xs space-y-1.5 transition-colors hover:border-border"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-foreground font-mono">{cat.model}</span>
                    <span className={cn('text-[10px] px-1.5 py-0.2 rounded border font-medium', colors.bg, colors.text, colors.border)}>
                      {cat.vendor}
                    </span>
                  </div>
                  <div className="text-xs font-mono font-medium text-emerald-500 dark:text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded">
                    {cat.effective_rate_display}
                  </div>
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    {cat.description}
                  </p>
                </div>
              )
            })}
          </CardContent>
        </Card>
      </div>

      {/* Daily Spend Timeline (if available) */}
      {timeline.length > 0 && (
        <Card className="border-border/60 bg-card/60 shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-semibold">Spend Timeline</CardTitle>
            <CardDescription className="text-xs">
              Daily expenditure trend across providers.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {timeline.slice(-7).map((point) => (
                <div key={point.date} className="flex items-center gap-3 text-xs">
                  <span className="font-mono text-muted-foreground w-24 shrink-0">{point.date}</span>
                  <div className="flex-1 bg-muted/40 rounded-full h-2 overflow-hidden flex">
                    {Object.entries(point.by_vendor).map(([vName, vCost]) => {
                      const pct = point.cost_usd > 0 ? (vCost / point.cost_usd) * 100 : 0
                      const colors = vendorColor(vName)
                      return (
                        <div
                          key={vName}
                          title={`${vName}: ${formatCostUsd(vCost)}`}
                          style={{ width: `${pct}%` }}
                          className={cn('h-full', colors.bar)}
                        />
                      )
                    })}
                  </div>
                  <span className="font-mono font-medium text-right w-20 shrink-0">
                    {formatCostUsd(point.cost_usd, 4)}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Live Request Ledger / Audit Trail */}
      <Card className="border-border/60 bg-card/60 shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base font-semibold">Inference Request Audit Ledger</CardTitle>
              <CardDescription className="text-xs">
                Append-only log of every external AI API call with latency, tokens, and micro-dollar cost.
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground font-mono">
                {requests ? `${requests.total} records` : '--'}
              </span>
            </div>
          </div>

          {/* Filter Toolbar */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 pt-3">
            {/* Search */}
            <div className="relative sm:col-span-2">
              <RiSearchLine className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
              <Input
                placeholder="Search route, model, summary..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value)
                  setPage(0)
                }}
                className="pl-8 text-xs h-9 bg-background/50"
              />
            </div>

            {/* Vendor filter */}
            <div>
              <select
                value={selectedVendor}
                onChange={(e) => {
                  setSelectedVendor(e.target.value)
                  setPage(0)
                }}
                className="w-full text-xs h-9 rounded-md border border-input bg-background/50 px-2.5 py-1 text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              >
                <option value="all">All Vendors</option>
                <option value="ElevenLabs">ElevenLabs</option>
                <option value="OpenRouter">OpenRouter</option>
                <option value="Google Cloud Vertex AI">Vertex AI</option>
              </select>
            </div>

            {/* Status filter */}
            <div>
              <select
                value={selectedStatus}
                onChange={(e) => {
                  setSelectedStatus(e.target.value)
                  setPage(0)
                }}
                className="w-full text-xs h-9 rounded-md border border-input bg-background/50 px-2.5 py-1 text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              >
                <option value="all">All Statuses</option>
                <option value="succeeded">Succeeded</option>
                <option value="failed">Failed</option>
                <option value="dry_run">Dry Run</option>
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow className="border-border/50 hover:bg-transparent">
                <TableHead className="text-xs">Time (UTC)</TableHead>
                <TableHead className="text-xs">Route / Model</TableHead>
                <TableHead className="text-xs">Vendor</TableHead>
                <TableHead className="text-xs">Status</TableHead>
                <TableHead className="text-xs text-right">Tokens</TableHead>
                <TableHead className="text-xs text-right">Latency</TableHead>
                <TableHead className="text-xs text-right">Cost (USD)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loadingRequests && (!requests || requests.items.length === 0) ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-xs text-muted-foreground">
                    <Spinner className="size-4 inline mr-2 text-primary" />
                    Loading request records…
                  </TableCell>
                </TableRow>
              ) : !requests || requests.items.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-xs text-muted-foreground">
                    No requests match your filters.
                  </TableCell>
                </TableRow>
              ) : (
                requests.items.map((r: LlmRequestItem) => {
                  const colors = vendorColor(r.vendor)
                  const isExpanded = expandedRequestId === r.id
                  const timeFormatted = new Date(r.created_at).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  })
                  const dateFormatted = new Date(r.created_at).toLocaleDateString([], {
                    month: 'short',
                    day: 'numeric',
                  })

                  return (
                    <TableRow
                      key={r.id}
                      onClick={() => setExpandedRequestId(isExpanded ? null : r.id)}
                      className={cn('cursor-pointer border-border/40 hover:bg-muted/40 transition-colors', isExpanded && 'bg-muted/30')}
                    >
                      <TableCell className="font-mono text-xs whitespace-nowrap text-muted-foreground">
                        <span>{dateFormatted} {timeFormatted}</span>
                      </TableCell>
                      <TableCell>
                        <div className="font-mono text-xs font-medium text-foreground">{r.route}</div>
                        <div className="text-[10px] text-muted-foreground truncate max-w-[180px]">
                          {r.model || '--'}
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className={cn('text-[10px] font-medium px-1.5 py-0.5 rounded border', colors.bg, colors.text, colors.border)}>
                          {r.vendor}
                        </span>
                      </TableCell>
                      <TableCell>
                        {r.status === 'succeeded' && (
                          <Badge variant="outline" className="border-emerald-500/40 text-emerald-500 bg-emerald-500/10 text-[10px] py-0">
                            succeeded
                          </Badge>
                        )}
                        {r.status === 'failed' && (
                          <Badge variant="destructive" className="text-[10px] py-0">
                            failed
                          </Badge>
                        )}
                        {r.status === 'dry_run' && (
                          <Badge variant="outline" className="border-purple-500/40 text-purple-500 bg-purple-500/10 text-[10px] py-0">
                            dry run
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs text-muted-foreground">
                        {r.prompt_tokens || r.completion_tokens ? (
                          <span>{(r.prompt_tokens || 0) + (r.completion_tokens || 0)}</span>
                        ) : (
                          '--'
                        )}
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs text-muted-foreground">
                        {formatLatency(r.latency_ms)}
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs font-bold text-foreground">
                        {formatCostUsd(r.estimated_cost_usd, 6)}
                      </TableCell>
                    </TableRow>
                  )
                })
              )}
            </TableBody>
          </Table>

          {/* Expanded Request Details Modal / Banner */}
          {expandedRequestId !== null && requests && (
            (() => {
              const item = requests.items.find((x) => x.id === expandedRequestId)
              if (!item) return null
              return (
                <div className="p-4 border-t bg-muted/20 text-xs space-y-2">
                  <div className="flex items-center justify-between font-mono font-medium">
                    <span>Request #{item.id} Details</span>
                    <Button variant="ghost" size="sm" onClick={() => setExpandedRequestId(null)} className="h-6 text-[10px]">
                      Close
                    </Button>
                  </div>
                  {item.input_summary && (
                    <div>
                      <span className="text-muted-foreground block mb-0.5">Input Summary:</span>
                      <pre className="p-2 rounded bg-background border font-mono text-[11px] overflow-x-auto whitespace-pre-wrap">
                        {item.input_summary}
                      </pre>
                    </div>
                  )}
                  {item.error_message && (
                    <div>
                      <span className="text-destructive block font-medium mb-0.5">Error Message:</span>
                      <pre className="p-2 rounded bg-destructive/10 text-destructive border border-destructive/20 font-mono text-[11px] overflow-x-auto whitespace-pre-wrap">
                        {item.error_message}
                      </pre>
                    </div>
                  )}
                </div>
              )
            })()
          )}

          {/* Pagination Controls */}
          {requests && requests.total > pageSize && (
            <div className="flex items-center justify-between p-3 border-t text-xs text-muted-foreground">
              <span>
                Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, requests.total)} of {requests.total}
              </span>
              <div className="flex items-center gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  className="h-8 text-xs"
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={(page + 1) * pageSize >= requests.total}
                  onClick={() => setPage((p) => p + 1)}
                  className="h-8 text-xs"
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
