import { defaultCache } from '@serwist/next/worker';
import type { PrecacheEntry, SerwistGlobalConfig } from 'serwist';
import { Serwist } from 'serwist';

// 이것은 Serwist의 타입 정의입니다.
declare global {
  interface WorkerGlobalScope extends SerwistGlobalConfig {
    // `injectionPoint`를 변경하려면 이것을 변경하세요.
    // `injectionPoint`는 Service Worker가 자신을 등록하는 위치입니다.
    // 기본적으로 `self.__SW_MANIFEST`입니다.
    __SW_MANIFEST: (string | PrecacheEntry)[];
  }
}

declare const self: WorkerGlobalScope;

const serwist = new Serwist({
  precacheEntries: self.__SW_MANIFEST as PrecacheEntry[],
  skipWaiting: true,
  clientsClaim: true,
  navigationPreload: true,
  runtimeCaching: defaultCache,
});

serwist.addEventListeners();
