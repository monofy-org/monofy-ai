/* eslint-disable */
/* tslint:disable */
/*
 * ---------------------------------------------------------------
 * ## THIS FILE WAS GENERATED VIA SWAGGER-TYPESCRIPT-API        ##
 * ##                                                           ##
 * ## AUTHOR: acacode                                           ##
 * ## SOURCE: https://github.com/acacode/swagger-typescript-api ##
 * ---------------------------------------------------------------
 */

/** Body_pdf_rip_api_pdf_rip_post */
export interface BodyPdfRipApiPdfRipPost {
  /**
   * Pdf
   * @format binary
   */
  pdf: File;
}

/** CannyRequest */
export interface CannyRequest {
  /** Image */
  image: string;
  /**
   * Threshold1
   * @default 100
   */
  threshold1?: number;
  /**
   * Threshold2
   * @default 200
   */
  threshold2?: number;
  /** Width */
  width?: number;
  /** Height */
  height?: number;
  /**
   * Return Json
   * @default false
   */
  return_json?: boolean;
}

/** ChatCompletionRequest */
export interface ChatCompletionRequest {
  /** Messages */
  messages: any[];
  /**
   * Model
   * @default "local"
   */
  model?: string;
  /**
   * Temperature
   * @default 0.7
   */
  temperature?: number;
  /**
   * Top P
   * @default 0.9
   */
  top_p?: number;
  /**
   * Top K
   * @default 20
   */
  top_k?: number;
  /**
   * Max Emojis
   * @default 1
   */
  max_emojis?: number;
  /**
   * Max Tokens
   * @default 200
   */
  max_tokens?: number;
  /**
   * Frequency Penalty
   * @default 1.05
   */
  frequency_penalty?: number;
  /**
   * Presence Penalty
   * @default 0
   */
  presence_penalty?: number;
  /**
   * Stream
   * @default false
   */
  stream?: boolean;
  /** Context */
  context?: string;
  /**
   * Bot Name
   * @default "Assistant"
   */
  bot_name?: string;
  /**
   * User Name
   * @default "User"
   */
  user_name?: string;
}

/** DepthRequest */
export interface DepthRequest {
  /** Image */
  image: string;
  /**
   * Median Filter
   * @default 5
   */
  median_filter?: number;
  /**
   * Return Json
   * @default false
   */
  return_json?: boolean;
}

/** DetectRequest */
export interface DetectRequest {
  /** Image */
  image: string;
  /**
   * Threshold
   * @default 0.8
   */
  threshold?: number;
  /**
   * Return Image
   * @default false
   */
  return_image?: boolean;
}

/** HTTPValidationError */
export interface HTTPValidationError {
  /** Detail */
  detail?: ValidationError[];
}

/** Img2ModelLGMRequest */
export interface Img2ModelLGMRequest {
  /** Image */
  image: string;
  /**
   * Num Inference Steps
   * @default 40
   */
  num_inference_steps?: number;
  /**
   * Guidance Scale
   * @default 3
   */
  guidance_scale?: number;
  /**
   * Negative Prompt
   * @default ""
   */
  negative_prompt?: string;
  /**
   * Format
   * @default "ply"
   */
  format?: "glb" | "ply";
}

/** Img2ModelTSRRequest */
export interface Img2ModelTSRRequest {
  /** Image */
  image: string;
  /**
   * Format
   * @default "glb"
   */
  format?: "glb" | "obj";
  /**
   * Foreground Ratio
   * @default [0.85]
   */
  foreground_ratio?: number;
}

/** Img2TxtRequest */
export interface Img2TxtRequest {
  /**
   * Prompt
   * @default "Describe the image."
   */
  prompt?: string;
  /**
   * History
   * @default []
   */
  history?: object[];
  /** Image */
  image?: string;
}

/** Img2VidXTRequest */
export interface Img2VidXTRequest {
  /** Image */
  image?: string;
  /**
   * Motion Bucket
   * @default 31
   */
  motion_bucket?: number;
  /**
   * Num Inference Steps
   * @default 6
   */
  num_inference_steps?: number;
  /**
   * Width
   * @default 512
   */
  width?: number;
  /**
   * Height
   * @default 512
   */
  height?: number;
  /**
   * Fps
   * @default 12
   */
  fps?: number;
  /**
   * Num Frames
   * @default 24
   */
  num_frames?: number;
  /**
   * Noise
   * @default 0
   */
  noise?: number;
  /**
   * Interpolate Film
   * @default 1
   */
  interpolate_film?: number;
  /**
   * Interpolate Rife
   * @default false
   */
  interpolate_rife?: boolean;
  /**
   * Fast Interpolate
   * @default true
   */
  fast_interpolate?: boolean;
  /**
   * Seed
   * @default -1
   */
  seed?: number;
  /** Audio */
  audio?: string;
}

/** ImgExifRequest */
export interface ImgExifRequest {
  /** Image */
  image: string;
  /** Exif */
  exif?: string;
}

/** MusicGenRequest */
export interface MusicGenRequest {
  /** Prompt */
  prompt: string;
  /**
   * Duration
   * @default 10
   */
  duration?: number;
  /**
   * Temperature
   * @default 0.9
   */
  temperature?: number;
  /**
   * Guidance Scale
   * @default 6.5
   */
  guidance_scale?: number;
  /**
   * Format
   * @default "wav"
   */
  format?: string;
  /**
   * Seed
   * @default -1
   */
  seed?: number;
  /**
   * Top P
   * @default 0.95
   */
  top_p?: number;
  /**
   * Streaming
   * @default false
   */
  streaming?: boolean;
  /** Wav Bytes */
  wav_bytes?: string;
  /**
   * Loop
   * @default false
   */
  loop?: boolean;
}

/** RembgRequest */
export interface RembgRequest {
  /** Image */
  image: string;
  /** Width */
  width: number;
  /** Height */
  height: number;
}

/** SummaryRequest */
export interface SummaryRequest {
  /** Url */
  url: string;
  /**
   * Prompt
   * @default "Summarize the following text scraped from the web."
   */
  prompt?: string;
  /**
   * Max Response Tokens
   * @default 200
   */
  max_response_tokens?: number;
}

/** TTSRequest */
export interface TTSRequest {
  /** Text */
  text: string;
  /**
   * Language
   * @default "en"
   */
  language?: string;
  /**
   * Voice
   * @default "female1"
   */
  voice?: string;
  /**
   * Temperature
   * @default 0.75
   */
  temperature?: number;
  /**
   * Speed
   * @default 1
   */
  speed?: number;
}

/** Txt2ImgRequest */
export interface Txt2ImgRequest {
  /**
   * Prompt
   * @default ""
   */
  prompt?: string;
  /**
   * Negative Prompt
   * @default ""
   */
  negative_prompt?: string;
  /**
   * Width
   * @default 768
   */
  width?: number;
  /**
   * Height
   * @default 768
   */
  height?: number;
  /**
   * Guidance Scale
   * @default 3
   */
  guidance_scale?: number;
  /** Num Inference Steps */
  num_inference_steps?: number;
  /**
   * Seed
   * @default -1
   */
  seed?: number;
  /**
   * Model Index
   * @default 0
   */
  model_index?: number;
  /** Scheduler */
  scheduler?: string;
  /**
   * Nsfw
   * @default false
   */
  nsfw?: boolean;
  /** Face Prompt */
  face_prompt?: string;
  /**
   * Upscale
   * @default 0
   */
  upscale?: number;
  /**
   * Strength
   * @default 0.65
   */
  strength?: number;
  /**
   * Auto Lora
   * @default true
   */
  auto_lora?: boolean;
  /**
   * Freeu
   * @default false
   */
  freeu?: boolean;
  /**
   * Hi
   * @default false
   */
  hi?: boolean;
  /**
   * Return Json
   * @default false
   */
  return_json?: boolean;
  /** Image */
  image?: string;
  /**
   * Tiling
   * @default false
   */
  tiling?: boolean;
}

/** Txt2ModelAvatarRequest */
export interface Txt2ModelAvatarRequest {
  /** Prompt */
  prompt: string;
  /**
   * Negative Prompt
   * @default ""
   */
  negative_prompt?: string;
}

/** Txt2ModelShapERequest */
export interface Txt2ModelShapERequest {
  /** Prompt */
  prompt: string;
  /**
   * Guidance Scale
   * @default 15
   */
  guidance_scale?: number;
  /**
   * Num Inference Steps
   * @default 64
   */
  num_inference_steps?: number;
  /**
   * Format
   * @default "glb"
   */
  format?: "glb" | "ply" | "gif";
  /**
   * Frame Size
   * @default 256
   */
  frame_size?: number;
}

/** Txt2VidRequest */
export interface Txt2VidRequest {
  /**
   * Prompt
   * @default ""
   */
  prompt?: string;
  /**
   * Negative Prompt
   * @default ""
   */
  negative_prompt?: string;
  /**
   * Width
   * @default 512
   */
  width?: number;
  /**
   * Height
   * @default 512
   */
  height?: number;
  /**
   * Guidance Scale
   * @default 2
   */
  guidance_scale?: number;
  /**
   * Num Frames
   * @default 16
   */
  num_frames?: number;
  /**
   * Num Inference Steps
   * @default 6
   */
  num_inference_steps?: number;
  /**
   * Fps
   * @default 12
   */
  fps?: number;
  /**
   * Seed
   * @default -1
   */
  seed?: number;
  /**
   * Interpolate Film
   * @default 1
   */
  interpolate_film?: number;
  /**
   * Interpolate Rife
   * @default 1
   */
  interpolate_rife?: number;
  /**
   * Fast Interpolate
   * @default true
   */
  fast_interpolate?: boolean;
  /** Audio */
  audio?: string;
}

/** TxtPersonalityRequest */
export interface TxtPersonalityRequest {
  /**
   * Min Age
   * @default 18
   */
  min_age?: number;
  /**
   * Max Age
   * @default 80
   */
  max_age?: number;
  /**
   * Gender
   * @default "random"
   */
  gender?: "random" | "male" | "female";
  /**
   * Description
   * @default ""
   */
  description?: string;
}

/** ValidationError */
export interface ValidationError {
  /** Location */
  loc: (string | number)[];
  /** Message */
  msg: string;
  /** Error Type */
  type: string;
}

/** Vid2VidRequest */
export interface Vid2VidRequest {
  /** Video */
  video: string;
  /** Prompt */
  prompt: string;
  /**
   * Negative Prompt
   * @default ""
   */
  negative_prompt?: string;
  /**
   * Seed
   * @default -1
   */
  seed?: number;
  /**
   * Rows
   * @default 2
   */
  rows?: number;
  /**
   * Cols
   * @default 2
   */
  cols?: number;
}

/** VisionRequest */
export interface VisionRequest {
  /** Image */
  image: string;
  /**
   * Prompt
   * @default "Describe the image in a few words."
   */
  prompt?: string;
  /**
   * Seed
   * @default -1
   */
  seed?: number;
}

/** YouTubeCaptionsRequest */
export interface YouTubeCaptionsRequest {
  /** Url */
  url: string;
  /**
   * Prompt
   * @default "Your task is to give a concise summary (one to 3 sentences) of a YouTube video."
   */
  prompt?: string;
  /**
   * Summary
   * @default false
   */
  summary?: boolean;
  /**
   * Max Response Tokens
   * @default 3000
   */
  max_response_tokens?: number;
}

/** YouTubeDownloadRequest */
export interface YouTubeDownloadRequest {
  /** Url */
  url: string;
  /**
   * Audio Only
   * @default false
   */
  audio_only?: boolean;
  /**
   * Start Time
   * @default 0
   */
  start_time?: number;
  /** Length */
  length?: number;
  /**
   * Format
   * @default "mp4"
   */
  format?: "mp4" | "gif";
  /**
   * Fps
   * @default 10
   */
  fps?: number;
  /** Text */
  text?: string;
  /** Width */
  width?: number;
}

/** YouTubeFramesRequest */
export interface YouTubeFramesRequest {
  /** Url */
  url: string;
  /**
   * Num Frames
   * @default 10
   */
  num_frames?: number;
  /**
   * Trim Start
   * @default 2
   */
  trim_start?: number;
  /**
   * Trim End
   * @default 2
   */
  trim_end?: number;
  /**
   * Summary
   * @default false
   */
  summary?: boolean;
  /**
   * Captions
   * @default false
   */
  captions?: boolean;
}

/** YouTubeGridRequest */
export interface YouTubeGridRequest {
  /** Url */
  url: string;
  /**
   * Rows
   * @default 3
   */
  rows?: number;
  /**
   * Cols
   * @default 3
   */
  cols?: number;
}

export type QueryParamsType = Record<string | number, any>;
export type ResponseFormat = keyof Omit<Body, "body" | "bodyUsed">;

export interface FullRequestParams extends Omit<RequestInit, "body"> {
  /** set parameter to `true` for call `securityWorker` for this request */
  secure?: boolean;
  /** request path */
  path: string;
  /** content type of request body */
  type?: ContentType;
  /** query params */
  query?: QueryParamsType;
  /** format of response (i.e. response.json() -> format: "json") */
  format?: ResponseFormat;
  /** request body */
  body?: unknown;
  /** base url */
  baseUrl?: string;
  /** request cancellation token */
  cancelToken?: CancelToken;
}

export type RequestParams = Omit<FullRequestParams, "body" | "method" | "query" | "path">;

export interface ApiConfig<SecurityDataType = unknown> {
  baseUrl?: string;
  baseApiParams?: Omit<RequestParams, "baseUrl" | "cancelToken" | "signal">;
  securityWorker?: (securityData: SecurityDataType | null) => Promise<RequestParams | void> | RequestParams | void;
  customFetch?: typeof fetch;
}

export interface HttpResponse<D extends unknown, E extends unknown = unknown> extends Response {
  data: D;
  error: E;
}

type CancelToken = Symbol | string | number;

export enum ContentType {
  Json = "application/json",
  FormData = "multipart/form-data",
  UrlEncoded = "application/x-www-form-urlencoded",
  Text = "text/plain",
}

export class HttpClient<SecurityDataType = unknown> {
  public baseUrl: string = "";
  private securityData: SecurityDataType | null = null;
  private securityWorker?: ApiConfig<SecurityDataType>["securityWorker"];
  private abortControllers = new Map<CancelToken, AbortController>();
  private customFetch = (...fetchParams: Parameters<typeof fetch>) => fetch(...fetchParams);

  private baseApiParams: RequestParams = {
    credentials: "same-origin",
    headers: {},
    redirect: "follow",
    referrerPolicy: "no-referrer",
  };

  constructor(apiConfig: ApiConfig<SecurityDataType> = {}) {
    Object.assign(this, apiConfig);
  }

  public setSecurityData = (data: SecurityDataType | null) => {
    this.securityData = data;
  };

  protected encodeQueryParam(key: string, value: any) {
    const encodedKey = encodeURIComponent(key);
    return `${encodedKey}=${encodeURIComponent(typeof value === "number" ? value : `${value}`)}`;
  }

  protected addQueryParam(query: QueryParamsType, key: string) {
    return this.encodeQueryParam(key, query[key]);
  }

  protected addArrayQueryParam(query: QueryParamsType, key: string) {
    const value = query[key];
    return value.map((v: any) => this.encodeQueryParam(key, v)).join("&");
  }

  protected toQueryString(rawQuery?: QueryParamsType): string {
    const query = rawQuery || {};
    const keys = Object.keys(query).filter((key) => "undefined" !== typeof query[key]);
    return keys
      .map((key) => (Array.isArray(query[key]) ? this.addArrayQueryParam(query, key) : this.addQueryParam(query, key)))
      .join("&");
  }

  protected addQueryParams(rawQuery?: QueryParamsType): string {
    const queryString = this.toQueryString(rawQuery);
    return queryString ? `?${queryString}` : "";
  }

  private contentFormatters: Record<ContentType, (input: any) => any> = {
    [ContentType.Json]: (input: any) =>
      input !== null && (typeof input === "object" || typeof input === "string") ? JSON.stringify(input) : input,
    [ContentType.Text]: (input: any) => (input !== null && typeof input !== "string" ? JSON.stringify(input) : input),
    [ContentType.FormData]: (input: any) =>
      Object.keys(input || {}).reduce((formData, key) => {
        const property = input[key];
        formData.append(
          key,
          property instanceof Blob
            ? property
            : typeof property === "object" && property !== null
            ? JSON.stringify(property)
            : `${property}`,
        );
        return formData;
      }, new FormData()),
    [ContentType.UrlEncoded]: (input: any) => this.toQueryString(input),
  };

  protected mergeRequestParams(params1: RequestParams, params2?: RequestParams): RequestParams {
    return {
      ...this.baseApiParams,
      ...params1,
      ...(params2 || {}),
      headers: {
        ...(this.baseApiParams.headers || {}),
        ...(params1.headers || {}),
        ...((params2 && params2.headers) || {}),
      },
    };
  }

  protected createAbortSignal = (cancelToken: CancelToken): AbortSignal | undefined => {
    if (this.abortControllers.has(cancelToken)) {
      const abortController = this.abortControllers.get(cancelToken);
      if (abortController) {
        return abortController.signal;
      }
      return void 0;
    }

    const abortController = new AbortController();
    this.abortControllers.set(cancelToken, abortController);
    return abortController.signal;
  };

  public abortRequest = (cancelToken: CancelToken) => {
    const abortController = this.abortControllers.get(cancelToken);

    if (abortController) {
      abortController.abort();
      this.abortControllers.delete(cancelToken);
    }
  };

  public request = async <T = any, E = any>({
    body,
    secure,
    path,
    type,
    query,
    format,
    baseUrl,
    cancelToken,
    ...params
  }: FullRequestParams): Promise<HttpResponse<T, E>> => {
    const secureParams =
      ((typeof secure === "boolean" ? secure : this.baseApiParams.secure) &&
        this.securityWorker &&
        (await this.securityWorker(this.securityData))) ||
      {};
    const requestParams = this.mergeRequestParams(params, secureParams);
    const queryString = query && this.toQueryString(query);
    const payloadFormatter = this.contentFormatters[type || ContentType.Json];
    const responseFormat = format || requestParams.format;

    return this.customFetch(`${baseUrl || this.baseUrl || ""}${path}${queryString ? `?${queryString}` : ""}`, {
      ...requestParams,
      headers: {
        ...(requestParams.headers || {}),
        ...(type && type !== ContentType.FormData ? { "Content-Type": type } : {}),
      },
      signal: (cancelToken ? this.createAbortSignal(cancelToken) : requestParams.signal) || null,
      body: typeof body === "undefined" || body === null ? null : payloadFormatter(body),
    }).then(async (response) => {
      const r = response as HttpResponse<T, E>;
      r.data = null as unknown as T;
      r.error = null as unknown as E;

      const data = !responseFormat
        ? r
        : await response[responseFormat]()
            .then((data) => {
              if (r.ok) {
                r.data = data;
              } else {
                r.error = data;
              }
              return r;
            })
            .catch((e) => {
              r.error = e;
              return r;
            });

      if (cancelToken) {
        this.abortControllers.delete(cancelToken);
      }

      if (!response.ok) throw data;
      return data;
    });
  };
}

/**
 * @title monofy-ai
 * @version 0.0.2
 *
 * Simple and multifaceted API for AI
 */
export class Api<SecurityDataType extends unknown> extends HttpClient<SecurityDataType> {
  api = {
    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgGetApiTxt2ImgGet
     * @summary Txt2Img Get
     * @request GET:/api/txt2img
     */
    txt2ImgGetApiTxt2ImgGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgApiTxt2ImgPost
     * @summary Txt2Img
     * @request POST:/api/txt2img
     */
    txt2ImgApiTxt2ImgPost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Img2ImgFromUrlApiImg2ImgGet
     * @summary Img2Img From Url
     * @request GET:/api/img2img
     */
    img2ImgFromUrlApiImg2ImgGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img2img`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Img2ImgApiImg2ImgPost
     * @summary Img2Img
     * @request POST:/api/img2img
     */
    img2ImgApiImg2ImgPost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img2img`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name InpaintFromUrlApiInpaintGet
     * @summary Inpaint From Url
     * @request GET:/api/inpaint
     */
    inpaintFromUrlApiInpaintGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/inpaint`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name InpaintApiInpaintPost
     * @summary Inpaint
     * @request POST:/api/inpaint
     */
    inpaintApiInpaintPost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/inpaint`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgFromUrlApiTxt2ImgCannyGet
     * @summary Txt2Img From Url
     * @request GET:/api/txt2img/canny
     */
    txt2ImgFromUrlApiTxt2ImgCannyGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/canny`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgApiTxt2ImgCannyPost
     * @summary Txt2Img
     * @request POST:/api/txt2img/canny
     */
    txt2ImgApiTxt2ImgCannyPost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/canny`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgFromUrlApiTxt2ImgDepthGet
     * @summary Txt2Img From Url
     * @request GET:/api/txt2img/depth
     */
    txt2ImgFromUrlApiTxt2ImgDepthGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/depth`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgApiTxt2ImgDepthPost
     * @summary Txt2Img
     * @request POST:/api/txt2img/depth
     */
    txt2ImgApiTxt2ImgDepthPost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/depth`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgInstantidFromUrlApiTxt2ImgInstantidGet
     * @summary Txt2Img Instantid From Url
     * @request GET:/api/txt2img/instantid
     */
    txt2ImgInstantidFromUrlApiTxt2ImgInstantidGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/instantid`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgInstantidApiTxt2ImgInstantidPost
     * @summary Txt2Img Instantid
     * @request POST:/api/txt2img/instantid
     */
    txt2ImgInstantidApiTxt2ImgInstantidPost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/instantid`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgCascadeFromUrlApiTxt2ImgCascadeGet
     * @summary Txt2Img Cascade From Url
     * @request GET:/api/txt2img/cascade
     */
    txt2ImgCascadeFromUrlApiTxt2ImgCascadeGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/cascade`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgCascadeApiTxt2ImgCascadePost
     * @summary Txt2Img Cascade
     * @request POST:/api/txt2img/cascade
     */
    txt2ImgCascadeApiTxt2ImgCascadePost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/cascade`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgFromUrlApiTxt2ImgControlnetGet
     * @summary Txt2Img From Url
     * @request GET:/api/txt2img/controlnet
     */
    txt2ImgFromUrlApiTxt2ImgControlnetGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<void, HTTPValidationError>({
        path: `/api/txt2img/controlnet`,
        method: "GET",
        query: query,
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgApiTxt2ImgControlnetPost
     * @summary Txt2Img
     * @request POST:/api/txt2img/controlnet
     */
    txt2ImgApiTxt2ImgControlnetPost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/txt2img/controlnet`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @tags Video Generation (text-to-video)
     * @name Txt2VidGetApiTxt2VidAnimateGet
     * @summary Txt2Vid Get
     * @request GET:/api/txt2vid/animate
     */
    txt2VidGetApiTxt2VidAnimateGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 512
         */
        width?: number;
        /**
         * Height
         * @default 512
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 2
         */
        guidance_scale?: number;
        /**
         * Num Frames
         * @default 16
         */
        num_frames?: number;
        /**
         * Num Inference Steps
         * @default 6
         */
        num_inference_steps?: number;
        /**
         * Fps
         * @default 12
         */
        fps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Interpolate Film
         * @default 1
         */
        interpolate_film?: number;
        /**
         * Interpolate Rife
         * @default 1
         */
        interpolate_rife?: number;
        /**
         * Fast Interpolate
         * @default true
         */
        fast_interpolate?: boolean;
        /** Audio */
        audio?: string;
      },
      params: RequestParams = {},
    ) =>
      this.request<void, HTTPValidationError>({
        path: `/api/txt2vid/animate`,
        method: "GET",
        query: query,
        ...params,
      }),

    /**
     * No description
     *
     * @tags Video Generation (text-to-video)
     * @name Txt2VidApiTxt2VidAnimatePost
     * @summary Txt2Vid
     * @request POST:/api/txt2vid/animate
     */
    txt2VidApiTxt2VidAnimatePost: (data: Txt2VidRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/txt2vid/animate`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @tags Video Generation (text-to-video)
     * @name Txt2VidGetApiTxt2VidZeroscopeGet
     * @summary Txt2Vid Get
     * @request GET:/api/txt2vid/zeroscope
     */
    txt2VidGetApiTxt2VidZeroscopeGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 512
         */
        width?: number;
        /**
         * Height
         * @default 512
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 2
         */
        guidance_scale?: number;
        /**
         * Num Frames
         * @default 16
         */
        num_frames?: number;
        /**
         * Num Inference Steps
         * @default 6
         */
        num_inference_steps?: number;
        /**
         * Fps
         * @default 12
         */
        fps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Interpolate Film
         * @default 1
         */
        interpolate_film?: number;
        /**
         * Interpolate Rife
         * @default 1
         */
        interpolate_rife?: number;
        /**
         * Fast Interpolate
         * @default true
         */
        fast_interpolate?: boolean;
        /** Audio */
        audio?: string;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2vid/zeroscope`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Video Generation (text-to-video)
     * @name Txt2VidApiTxt2VidZeroscopePost
     * @summary Txt2Vid
     * @request POST:/api/txt2vid/zeroscope
     */
    txt2VidApiTxt2VidZeroscopePost: (data: Txt2VidRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2vid/zeroscope`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @name Img2VidFromUrlApiImg2VidXtGet
     * @summary Img2Vid From Url
     * @request GET:/api/img2vid/xt
     */
    img2VidFromUrlApiImg2VidXtGet: (
      query?: {
        /** Image */
        image?: string;
        /**
         * Motion Bucket
         * @default 31
         */
        motion_bucket?: number;
        /**
         * Num Inference Steps
         * @default 6
         */
        num_inference_steps?: number;
        /**
         * Width
         * @default 512
         */
        width?: number;
        /**
         * Height
         * @default 512
         */
        height?: number;
        /**
         * Fps
         * @default 12
         */
        fps?: number;
        /**
         * Num Frames
         * @default 24
         */
        num_frames?: number;
        /**
         * Noise
         * @default 0
         */
        noise?: number;
        /**
         * Interpolate Film
         * @default 1
         */
        interpolate_film?: number;
        /**
         * Interpolate Rife
         * @default false
         */
        interpolate_rife?: boolean;
        /**
         * Fast Interpolate
         * @default true
         */
        fast_interpolate?: boolean;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /** Audio */
        audio?: string;
      },
      params: RequestParams = {},
    ) =>
      this.request<void, HTTPValidationError>({
        path: `/api/img2vid/xt`,
        method: "GET",
        query: query,
        ...params,
      }),

    /**
     * No description
     *
     * @name Img2VidApiImg2VidXtPost
     * @summary Img2Vid
     * @request POST:/api/img2vid/xt
     */
    img2VidApiImg2VidXtPost: (data: Img2VidXTRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/img2vid/xt`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @tags Video Generation (text-to-video)
     * @name Txt2VidFromUrlApiTxt2VidZeroGet
     * @summary Txt2Vid From Url
     * @request GET:/api/txt2vid/zero
     */
    txt2VidFromUrlApiTxt2VidZeroGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 512
         */
        width?: number;
        /**
         * Height
         * @default 512
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 2
         */
        guidance_scale?: number;
        /**
         * Num Frames
         * @default 16
         */
        num_frames?: number;
        /**
         * Num Inference Steps
         * @default 6
         */
        num_inference_steps?: number;
        /**
         * Fps
         * @default 12
         */
        fps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Interpolate Film
         * @default 1
         */
        interpolate_film?: number;
        /**
         * Interpolate Rife
         * @default 1
         */
        interpolate_rife?: number;
        /**
         * Fast Interpolate
         * @default true
         */
        fast_interpolate?: boolean;
        /** Audio */
        audio?: string;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2vid/zero`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Video Generation (text-to-video)
     * @name Txt2VidApiTxt2VidZeroPost
     * @summary Txt2Vid
     * @request POST:/api/txt2vid/zero
     */
    txt2VidApiTxt2VidZeroPost: (data: Txt2VidRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2vid/zero`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Processing
     * @name DepthEstimationFromUrlApiImgDepthGet
     * @summary Depth Estimation From Url
     * @request GET:/api/img/depth
     */
    depthEstimationFromUrlApiImgDepthGet: (
      query: {
        /** Image */
        image: string;
        /**
         * Median Filter
         * @default 5
         */
        median_filter?: number;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img/depth`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Processing
     * @name DepthEstimationApiImgDepthPost
     * @summary Depth Estimation
     * @request POST:/api/img/depth
     */
    depthEstimationApiImgDepthPost: (data: DepthRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img/depth`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Processing
     * @name DepthEstimationFromUrlApiImgDepthMidasGet
     * @summary Depth Estimation From Url
     * @request GET:/api/img/depth/midas
     */
    depthEstimationFromUrlApiImgDepthMidasGet: (
      query: {
        /** Image */
        image: string;
        /**
         * Median Filter
         * @default 5
         */
        median_filter?: number;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<void, HTTPValidationError>({
        path: `/api/img/depth/midas`,
        method: "GET",
        query: query,
        ...params,
      }),

    /**
     * @description API route for depth detection
     *
     * @tags Image Processing
     * @name DepthEstimationApiImgDepthMidasPost
     * @summary Depth Estimation
     * @request POST:/api/img/depth/midas
     */
    depthEstimationApiImgDepthMidasPost: (data: DepthRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/img/depth/midas`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @name DetectFromUrlApiDetectYolosGet
     * @summary Detect From Url
     * @request GET:/api/detect/yolos
     */
    detectFromUrlApiDetectYolosGet: (
      query: {
        /** Image */
        image: string;
        /**
         * Threshold
         * @default 0.8
         */
        threshold?: number;
        /**
         * Return Image
         * @default false
         */
        return_image?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/detect/yolos`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @name DetectYolosApiDetectYolosPost
     * @summary Detect Yolos
     * @request POST:/api/detect/yolos
     */
    detectYolosApiDetectYolosPost: (data: DetectRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/detect/yolos`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags 3D Model Generation
     * @name ShapeGetApiTxt2ModelShapeGet
     * @summary Shape Get
     * @request GET:/api/txt2model/shape
     */
    shapeGetApiTxt2ModelShapeGet: (data: Txt2ModelShapERequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2model/shape`,
        method: "GET",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags 3D Model Generation
     * @name ShapeApiTxt2ModelShapePost
     * @summary Shape
     * @request POST:/api/txt2model/shape
     */
    shapeApiTxt2ModelShapePost: (data: Txt2ModelShapERequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2model/shape`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags 3D Model Generation
     * @name Img2ModelFromUrlApiImg2ModelLgmGet
     * @summary Img2Model From Url
     * @request GET:/api/img2model/lgm
     */
    img2ModelFromUrlApiImg2ModelLgmGet: (
      query: {
        /** Image */
        image: string;
        /**
         * Num Inference Steps
         * @default 40
         */
        num_inference_steps?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Format
         * @default "ply"
         */
        format?: "glb" | "ply";
      },
      params: RequestParams = {},
    ) =>
      this.request<void, HTTPValidationError>({
        path: `/api/img2model/lgm`,
        method: "GET",
        query: query,
        ...params,
      }),

    /**
     * No description
     *
     * @tags 3D Model Generation
     * @name Img2ModelApiImg2ModelLgmPost
     * @summary Img2Model
     * @request POST:/api/img2model/lgm
     */
    img2ModelApiImg2ModelLgmPost: (data: Img2ModelLGMRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/img2model/lgm`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @tags 3D Model Generation
     * @name Img2ModelFromUrlApiImg2ModelTsrGet
     * @summary Img2Model From Url
     * @request GET:/api/img2model/tsr
     */
    img2ModelFromUrlApiImg2ModelTsrGet: (
      query: {
        /** Image */
        image: string;
        /**
         * Format
         * @default "glb"
         */
        format?: "glb" | "obj";
        /**
         * Foreground Ratio
         * @default [0.85]
         */
        foreground_ratio?: number;
      },
      params: RequestParams = {},
    ) =>
      this.request<void, HTTPValidationError>({
        path: `/api/img2model/tsr`,
        method: "GET",
        query: query,
        ...params,
      }),

    /**
     * No description
     *
     * @tags 3D Model Generation
     * @name Img2ModelApiImg2ModelTsrPost
     * @summary Img2Model
     * @request POST:/api/img2model/tsr
     */
    img2ModelApiImg2ModelTsrPost: (data: Img2ModelTSRRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/img2model/tsr`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @name RemoveBackgroundApiRembgPost
     * @summary Remove Background
     * @request POST:/api/rembg
     */
    removeBackgroundApiRembgPost: (data: RembgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/rembg`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @name VisionFromUrlApiVisionGet
     * @summary Vision From Url
     * @request GET:/api/vision
     */
    visionFromUrlApiVisionGet: (
      query: {
        /** Image */
        image: string;
        /**
         * Prompt
         * @default "Describe the image in a few words."
         */
        prompt?: string;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/vision`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @name VisionApiVisionPost
     * @summary Vision
     * @request POST:/api/vision
     */
    visionApiVisionPost: (data: VisionRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/vision`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image-to-Text
     * @name Img2TxtFromUrlApiImg2TxtLlavaGet
     * @summary Img2Txt From Url
     * @request GET:/api/img2txt/llava
     */
    img2TxtFromUrlApiImg2TxtLlavaGet: (
      data: object[],
      query?: {
        /**
         * Prompt
         * @default "Describe the image."
         */
        prompt?: string;
        /** Image */
        image?: string;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img2txt/llava`,
        method: "GET",
        query: query,
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image-to-Text
     * @name Img2TxtApiImg2TxtLlavaPost
     * @summary Img2Txt
     * @request POST:/api/img2txt/llava
     */
    img2TxtApiImg2TxtLlavaPost: (data: Img2TxtRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img2txt/llava`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @name MusicgenGetApiMusicgenGet
     * @summary Musicgen Get
     * @request GET:/api/musicgen
     */
    musicgenGetApiMusicgenGet: (
      query: {
        /** Prompt */
        prompt: string;
        /**
         * Duration
         * @default 10
         */
        duration?: number;
        /**
         * Temperature
         * @default 0.9
         */
        temperature?: number;
        /**
         * Guidance Scale
         * @default 6.5
         */
        guidance_scale?: number;
        /**
         * Format
         * @default "wav"
         */
        format?: string;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Top P
         * @default 0.95
         */
        top_p?: number;
        /**
         * Streaming
         * @default false
         */
        streaming?: boolean;
        /** Wav Bytes */
        wav_bytes?: string;
        /**
         * Loop
         * @default false
         */
        loop?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<void, HTTPValidationError>({
        path: `/api/musicgen`,
        method: "GET",
        query: query,
        ...params,
      }),

    /**
     * No description
     *
     * @name MusicgenApiMusicgenPost
     * @summary Musicgen
     * @request POST:/api/musicgen
     */
    musicgenApiMusicgenPost: (data: MusicGenRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/musicgen`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @name ChatCompletionsApiChatCompletionsPost
     * @summary Chat Completions
     * @request POST:/api/chat/completions
     */
    chatCompletionsApiChatCompletionsPost: (data: ChatCompletionRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/chat/completions`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @name ChatStreamingPostApiChatStreamPost
     * @summary Chat Streaming Post
     * @request POST:/api/chat/stream
     */
    chatStreamingPostApiChatStreamPost: (data: ChatCompletionRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/chat/stream`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ModelAvatarApiTxt2ModelAvatarPost
     * @summary Txt2Model Avatar
     * @request POST:/api/txt2model/avatar
     */
    txt2ModelAvatarApiTxt2ModelAvatarPost: (data: Txt2ModelAvatarRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2model/avatar`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ModelAvatarGenerateApiTxt2ModelAvatarGeneratePost
     * @summary Txt2Model Avatar Generate
     * @request POST:/api/txt2model/avatar/generate
     */
    txt2ModelAvatarGenerateApiTxt2ModelAvatarGeneratePost: (
      query: {
        /** Prompt */
        prompt: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2model/avatar/generate`,
        method: "POST",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Text-to-Speech (TTS)
     * @name TtsGetApiTtsGet
     * @summary Tts Get
     * @request GET:/api/tts
     */
    ttsGetApiTtsGet: (
      query: {
        /** Text */
        text: string;
        /**
         * Language
         * @default "en"
         */
        language?: string;
        /**
         * Voice
         * @default "female1"
         */
        voice?: string;
        /**
         * Temperature
         * @default 0.75
         */
        temperature?: number;
        /**
         * Speed
         * @default 1
         */
        speed?: number;
      },
      params: RequestParams = {},
    ) =>
      this.request<void, HTTPValidationError>({
        path: `/api/tts`,
        method: "GET",
        query: query,
        ...params,
      }),

    /**
     * No description
     *
     * @tags Text-to-Speech (TTS)
     * @name TtsApiTtsPost
     * @summary Tts
     * @request POST:/api/tts
     */
    ttsApiTtsPost: (data: TTSRequest, params: RequestParams = {}) =>
      this.request<void, HTTPValidationError>({
        path: `/api/tts`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        ...params,
      }),

    /**
     * No description
     *
     * @tags Text Generation
     * @name TxtSummaryFromUrlApiTxtSummaryGet
     * @summary Txt Summary From Url
     * @request GET:/api/txt/summary
     */
    txtSummaryFromUrlApiTxtSummaryGet: (
      query: {
        /** Url */
        url: string;
        /**
         * Prompt
         * @default "Summarize the following text scraped from the web."
         */
        prompt?: string;
        /**
         * Max Response Tokens
         * @default 200
         */
        max_response_tokens?: number;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt/summary`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Text Generation
     * @name TxtSummaryApiTxtSummaryPost
     * @summary Txt Summary
     * @request POST:/api/txt/summary
     */
    txtSummaryApiTxtSummaryPost: (data: SummaryRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt/summary`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags YouTube Tools
     * @name DownloadYoutubeVideoFromUrlApiYoutubeDownloadGet
     * @summary Download Youtube Video From Url
     * @request GET:/api/youtube/download
     */
    downloadYoutubeVideoFromUrlApiYoutubeDownloadGet: (
      query: {
        /** Url */
        url: string;
        /**
         * Audio Only
         * @default false
         */
        audio_only?: boolean;
        /**
         * Start Time
         * @default 0
         */
        start_time?: number;
        /** Length */
        length?: number;
        /**
         * Format
         * @default "mp4"
         */
        format?: "mp4" | "gif";
        /**
         * Fps
         * @default 10
         */
        fps?: number;
        /** Text */
        text?: string;
        /** Width */
        width?: number;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/youtube/download`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags YouTube Tools
     * @name DownloadYoutubeVideoApiYoutubeDownloadPost
     * @summary Download Youtube Video
     * @request POST:/api/youtube/download
     */
    downloadYoutubeVideoApiYoutubeDownloadPost: (data: YouTubeDownloadRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/youtube/download`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags YouTube Tools
     * @name CaptionsFromUrlApiYoutubeCaptionsGet
     * @summary Captions From Url
     * @request GET:/api/youtube/captions
     */
    captionsFromUrlApiYoutubeCaptionsGet: (
      query: {
        /** Url */
        url: string;
        /**
         * Prompt
         * @default "Your task is to give a concise summary (one to 3 sentences) of a YouTube video."
         */
        prompt?: string;
        /**
         * Summary
         * @default false
         */
        summary?: boolean;
        /**
         * Max Response Tokens
         * @default 3000
         */
        max_response_tokens?: number;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/youtube/captions`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags YouTube Tools
     * @name CaptionsApiYoutubeCaptionsPost
     * @summary Captions
     * @request POST:/api/youtube/captions
     */
    captionsApiYoutubeCaptionsPost: (data: YouTubeCaptionsRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/youtube/captions`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags YouTube Tools
     * @name YoutubeGridFromUrlApiYoutubeGridGet
     * @summary Youtube Grid From Url
     * @request GET:/api/youtube/grid
     */
    youtubeGridFromUrlApiYoutubeGridGet: (
      query: {
        /** Url */
        url: string;
        /**
         * Rows
         * @default 3
         */
        rows?: number;
        /**
         * Cols
         * @default 3
         */
        cols?: number;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/youtube/grid`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags YouTube Tools
     * @name YoutubeGridApiYoutubeGridPost
     * @summary Youtube Grid
     * @request POST:/api/youtube/grid
     */
    youtubeGridApiYoutubeGridPost: (data: YouTubeGridRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/youtube/grid`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags YouTube Tools
     * @name YoutubeFramesFromUrlApiYoutubeFramesGet
     * @summary Youtube Frames From Url
     * @request GET:/api/youtube/frames
     */
    youtubeFramesFromUrlApiYoutubeFramesGet: (
      query: {
        /** Url */
        url: string;
        /**
         * Num Frames
         * @default 10
         */
        num_frames?: number;
        /**
         * Trim Start
         * @default 2
         */
        trim_start?: number;
        /**
         * Trim End
         * @default 2
         */
        trim_end?: number;
        /**
         * Summary
         * @default false
         */
        summary?: boolean;
        /**
         * Captions
         * @default false
         */
        captions?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/youtube/frames`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags YouTube Tools
     * @name YoutubeFramesApiYoutubeFramesPost
     * @summary Youtube Frames
     * @request POST:/api/youtube/frames
     */
    youtubeFramesApiYoutubeFramesPost: (data: YouTubeFramesRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/youtube/frames`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Video Generation
     * @name Vid2VidFromUrlApiVid2VidGet
     * @summary Vid2Vid From Url
     * @request GET:/api/vid2vid
     */
    vid2VidFromUrlApiVid2VidGet: (
      query: {
        /** Video */
        video: string;
        /** Prompt */
        prompt: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Rows
         * @default 2
         */
        rows?: number;
        /**
         * Cols
         * @default 2
         */
        cols?: number;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/vid2vid`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Video Generation
     * @name Vid2VidApiVid2VidPost
     * @summary Vid2Vid
     * @request POST:/api/vid2vid
     */
    vid2VidApiVid2VidPost: (data: Vid2VidRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/vid2vid`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Img2ImgLoopbackApiImg2ImgLoopbackPost
     * @summary Img2Img Loopback
     * @request POST:/api/img2img/loopback
     */
    img2ImgLoopbackApiImg2ImgLoopbackPost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img2img/loopback`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Text Generation
     * @name PersonalityGetApiTxtProfileGet
     * @summary Personality Get
     * @request GET:/api/txt/profile
     */
    personalityGetApiTxtProfileGet: (
      query?: {
        /**
         * Min Age
         * @default 18
         */
        min_age?: number;
        /**
         * Max Age
         * @default 80
         */
        max_age?: number;
        /**
         * Gender
         * @default "random"
         */
        gender?: "random" | "male" | "female";
        /**
         * Description
         * @default ""
         */
        description?: string;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt/profile`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Text Generation
     * @name GeneratePersonalityApiTxtProfilePost
     * @summary Generate Personality
     * @request POST:/api/txt/profile
     */
    generatePersonalityApiTxtProfilePost: (data: TxtPersonalityRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt/profile`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgFaceGetApiTxt2ImgFaceGet
     * @summary Txt2Img Face Get
     * @request GET:/api/txt2img/face
     */
    txt2ImgFaceGetApiTxt2ImgFaceGet: (
      query?: {
        /**
         * Prompt
         * @default ""
         */
        prompt?: string;
        /**
         * Negative Prompt
         * @default ""
         */
        negative_prompt?: string;
        /**
         * Width
         * @default 768
         */
        width?: number;
        /**
         * Height
         * @default 768
         */
        height?: number;
        /**
         * Guidance Scale
         * @default 3
         */
        guidance_scale?: number;
        /** Num Inference Steps */
        num_inference_steps?: number;
        /**
         * Seed
         * @default -1
         */
        seed?: number;
        /**
         * Model Index
         * @default 0
         */
        model_index?: number;
        /** Scheduler */
        scheduler?: string;
        /**
         * Nsfw
         * @default false
         */
        nsfw?: boolean;
        /** Face Prompt */
        face_prompt?: string;
        /**
         * Upscale
         * @default 0
         */
        upscale?: number;
        /**
         * Strength
         * @default 0.65
         */
        strength?: number;
        /**
         * Auto Lora
         * @default true
         */
        auto_lora?: boolean;
        /**
         * Freeu
         * @default false
         */
        freeu?: boolean;
        /**
         * Hi
         * @default false
         */
        hi?: boolean;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
        /** Image */
        image?: string;
        /**
         * Tiling
         * @default false
         */
        tiling?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/face`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Generation
     * @name Txt2ImgFaceApiTxt2ImgFacePost
     * @summary Txt2Img Face
     * @request POST:/api/txt2img/face
     */
    txt2ImgFaceApiTxt2ImgFacePost: (data: Txt2ImgRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/txt2img/face`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Processing
     * @name CannyFromUrlApiImgCannyGet
     * @summary Canny From Url
     * @request GET:/api/img/canny
     */
    cannyFromUrlApiImgCannyGet: (
      query: {
        /** Image */
        image: string;
        /**
         * Threshold1
         * @default 100
         */
        threshold1?: number;
        /**
         * Threshold2
         * @default 200
         */
        threshold2?: number;
        /** Width */
        width?: number;
        /** Height */
        height?: number;
        /**
         * Return Json
         * @default false
         */
        return_json?: boolean;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img/canny`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image Processing
     * @name CannyApiImgCannyPost
     * @summary Canny
     * @request POST:/api/img/canny
     */
    cannyApiImgCannyPost: (data: CannyRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img/canny`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image EXIF Tools
     * @name ImgExifGetApiImgExifGet
     * @summary Img Exif Get
     * @request GET:/api/img/exif
     */
    imgExifGetApiImgExifGet: (
      query: {
        /** Image */
        image: string;
        /** Exif */
        exif?: string;
      },
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img/exif`,
        method: "GET",
        query: query,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags Image EXIF Tools
     * @name ImgExifApiImgExifPost
     * @summary Img Exif
     * @request POST:/api/img/exif
     */
    imgExifApiImgExifPost: (data: ImgExifRequest, params: RequestParams = {}) =>
      this.request<any, HTTPValidationError>({
        path: `/api/img/exif`,
        method: "POST",
        body: data,
        type: ContentType.Json,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @tags PDF
     * @name PdfRipApiPdfRipPost
     * @summary Pdf Rip
     * @request POST:/api/pdf/rip
     */
    pdfRipApiPdfRipPost: (
      query: {
        /** Pages */
        pages: string;
      },
      data: BodyPdfRipApiPdfRipPost,
      params: RequestParams = {},
    ) =>
      this.request<any, HTTPValidationError>({
        path: `/api/pdf/rip`,
        method: "POST",
        query: query,
        body: data,
        type: ContentType.FormData,
        format: "json",
        ...params,
      }),

    /**
     * No description
     *
     * @name GoogleTrendsApiGoogleTrendsGet
     * @summary Google Trends
     * @request GET:/api/google/trends
     */
    googleTrendsApiGoogleTrendsGet: (params: RequestParams = {}) =>
      this.request<any, any>({
        path: `/api/google/trends`,
        method: "GET",
        format: "json",
        ...params,
      }),
  };
}
