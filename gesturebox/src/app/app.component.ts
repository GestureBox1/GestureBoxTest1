import { Component, OnInit, Output, EventEmitter } from '@angular/core';
import { Router } from '@angular/router';
import { InitializeService } from '@kognifai/poseidon-ng-initialize-service';
import { SidebarsVisibilityService } from '@kognifai/poseidon-sidebar-visibilityservice';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
    initializing: boolean;

    constructor(
        private router: Router,
        private initializeService: InitializeService,
        private sidebarsVisibilityService: SidebarsVisibilityService
    ) {
        this.initializing = true;
    }

    ngOnInit() {
        // PWA:- Checking Network Status
        this.updateNetworkStatusUI();
        window.addEventListener('online', this.updateNetworkStatusUI);
        window.addEventListener('offline', this.updateNetworkStatusUI);

        this.initializeService.initialize().subscribe(
          () => { },
          error => { console.log('Initialize error.'); },
          () => {
              // finished
              this.initializing = false;
              this.router.initialNavigation();
          }
        );
    }

    // Checking App Status, whether its online or offline
    private updateNetworkStatusUI() {
        if (navigator.onLine) {
            // you might be online
            (document.getElementsByTagName('body') as any).style = '';
        } else {
            // 100% sure, you are offline
            (document.getElementsByTagName('body') as any).style = 'filter: grayscale(1)';
        }
    }
}
